import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import OneHotEncoder

from ctgan.constants import *

class DataTransformer(object):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar
    [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.

    Args:
        n_cluster (int):
            Number of modes.
        epsilon (float):
            Epsilon value.
    """

    def __init__(self, metadata, n_clusters=10, epsilon=0.005):
        self.n_clusters = n_clusters
        self.epsilon = epsilon

        self.metadata = metadata

        self.output_info = None
        self.output_dimensions = None
        self.dtypes = None
        self.fit_meta = None

    # @ignore_warnings(category=ConvergenceWarning)
    def _fit_continuous(self, col_name, data):
        gm = BayesianGaussianMixture(
            self.n_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            n_init=1
        )
        gm.fit(data)
        components = gm.weights_ > self.epsilon
        num_components = components.sum()

        return {
            'name': col_name,
            'model': gm,
            'components': components,
            'output_info': [(1, 'tanh'), (num_components, 'softmax')],
            'output_dimensions': 1 + num_components,
        }

    def _fit_discrete(self, col_name, categories):

        return {
            'name': col_name,
            'categories': categories,
            'output_info': [(len(categories), 'softmax')],
            'output_dimensions': len(categories)
        }

    def fit(self, data):
        self.output_info = []
        self.output_dimensions = 0

        self.dtypes = {}
        self.fit_meta = []

        if not isinstance(data, pd.DataFrame):
            self.dataframe = False
            data = pd.DataFrame(data)
        else:
            self.dataframe = True

        for col_info in self.metadata['columns']:
            col_name = col_info['name']
            col_type = col_info['type']
            column_data = data[col_name].values.reshape(-1, 1)

            if col_type == FLOAT:
                col_meta = self._fit_continuous(col_name, column_data)
                self.dtypes[col_name] = np.float

            elif col_type == INTEGER:
                col_meta = self._fit_continuous(col_name, column_data)
                self.dtypes[col_name] = np.int

            elif col_type == CATEGORICAL or col_type == ORDINAL:
                col_meta = self._fit_discrete(col_name, col_info['i2s'])
                self.dtypes[col_name] = np.object

            else:
                raise ValueError(f'Unknown dtype {col_type} for column {col_name}')

            self.output_info += col_meta['output_info']
            self.output_dimensions += col_meta['output_dimensions']

            self.fit_meta.append(col_meta)

    def _transform_continuous(self, column_meta, data):
        components = column_meta['components']
        model = column_meta['model']

        means = model.means_.reshape((1, self.n_clusters))
        stds = np.sqrt(model.covariances_).reshape((1, self.n_clusters))
        features = (data - means) / (4 * stds)

        probs = model.predict_proba(data)

        n_opts = components.sum()
        features = features[:, components]
        probs = probs[:, components]

        opt_sel = np.zeros(len(data), dtype='int')
        for i in range(len(data)):
            pp = probs[i] + 1e-6
            pp = pp / pp.sum()
            opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

        idx = np.arange((len(features)))
        features = features[idx, opt_sel].reshape([-1, 1])
        features = np.clip(features, -.99, .99)

        probs_onehot = np.zeros_like(probs)
        probs_onehot[np.arange(len(probs)), opt_sel] = 1
        return [features, probs_onehot]

    def _transform_discrete(self, column_meta, col_data):
        categories = column_meta['categories']
        data_ohe = self._one_hot(col_data, categories)

        return data_ohe

    def _one_hot(self, col_data, categories):
        col_data_onehot = np.zeros((len(col_data), len(categories)))
        cidx = [categories.index(c) for c in col_data]
        col_data_onehot[np.arange(len(col_data)), cidx] = 1

        return col_data_onehot

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        values = []
        for meta in self.fit_meta:
            column_data = data[[meta['name']]].values
            if 'model' in meta:
                values += self._transform_continuous(meta, column_data)
            else:
                values.append(self._transform_discrete(meta, column_data))

        return np.concatenate(values, axis=1).astype(float)

    def _inverse_transform_continuous(self, meta, data, sigma):
        model = meta['model']
        components = meta['components']

        u = data[:, 0]
        v = data[:, 1:]

        if sigma is not None:
            u = np.random.normal(u, sigma)

        u = np.clip(u, -1, 1)
        v_t = np.ones((len(data), self.n_clusters)) * -100
        v_t[:, components] = v
        v = v_t
        means = model.means_.reshape([-1])
        stds = np.sqrt(model.covariances_).reshape([-1])
        p_argmax = np.argmax(v, axis=1)
        std_t = stds[p_argmax]
        mean_t = means[p_argmax]
        column = u * 4 * std_t + mean_t

        return column

    def _inverse_transform_discrete(self, meta, oh_data):
        categories = meta['categories']
        data = self._reverse_one_hot(oh_data, categories)

        return data.reshape(-1, 1)

    def _reverse_one_hot(self, col_encoded, categories):
        cat_idx = np.argmax(col_encoded, axis=1)
        col_data = np.array([categories[i] for i in cat_idx])

        return col_data

    def inverse_transform(self, data, sigmas):
        start = 0
        output = []
        column_names = []
        for col_meta in self.fit_meta:
            print(col_meta['name'])
            dimensions = col_meta['output_dimensions']
            col_data = data[:, start:start + dimensions]

            if 'model' in col_meta:
                sigma = sigmas[start] if sigmas else None
                inverted = self._inverse_transform_continuous(col_meta, col_data, sigma)
            else:
                inverted = self._inverse_transform_discrete(col_meta, col_data)
            print(inverted[:10])
            output.append(inverted)
            column_names.append(col_meta['name'])
            start += dimensions

        output = np.column_stack(output)
        output = pd.DataFrame(output, columns=column_names).astype(self.dtypes)
        if not self.dataframe:
            output = output.values

        return output
