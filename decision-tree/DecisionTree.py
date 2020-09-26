class DecisionTree:

    def __init__(self, matrix, math, max_depth):
        self.__depth = 0
        self.__trees = {}
        self.__matrix = matrix
        self.__math = math
        self.__max_depth = max_depth

    def fit(self, x, y, par_node=None, depth=0):
        if par_node is None:
            par_node = {}
        if par_node is None:
            return None
        elif len(y) == 0:
            return None
        elif self.__all_same(y):
            return {'val': y[0]}
        elif depth >= self.__max_depth:
            return None
        else:
            col, cutoff, entropy = self.__find_best_split_of_all(x, y)
            y_left = y[x[:, col] < cutoff]
            y_right = y[x[:, col] >= cutoff]
            par_node = {
                'index_col': col,
                'cutoff': cutoff,
                'val': self.__math.round(self.__math.mean(y)),
                'left': self.fit(x[x[:, col] < cutoff], y_left, {}, depth + 1),
                'right': self.fit(x[x[:, col] >= cutoff], y_right, {}, depth + 1),
                'depth': depth
            }
            self.__depth += 1
            self.__trees = par_node
            return par_node

    def __find_best_split_of_all(self, x, y):
        col = None
        min_entropy = 1
        cutoff = None
        for i, c in enumerate(x.T):
            entropy, cur_cutoff = self.__find_best_split(c, y)
            if entropy == 0:
                return i, cur_cutoff, entropy
            elif entropy <= min_entropy:
                col = i
                min_entropy = entropy
                cutoff = cur_cutoff
        return col, cutoff, min_entropy

    def __entropy_function(self, c, n):
        return -(c * 1.0 / n) * self.__math.log2(c * 1.0 / n)

    def entropy_cal(self, c1, c2):
        if c1 == 0 or c2 == 0:
            return 0
        return self.__entropy_function(c1, c1 + c2) + self.__entropy_function(c2, c1 + c2)

    def entropy_of_one_division(self, division):
        s = 0
        n = len(division)
        classes = set(division.flatten())
        for c in classes:
            c1, c2 = self.__get_class_set(division, c)
            n_c = sum(c1)
            e = n_c * 1.0 / n * self.entropy_cal(sum(c1), sum(c2))  # weighted avg
            s += e
        return s, n

    @staticmethod
    def __get_class_set(division, c):
        return (division == c), (division != c)

    def get_entropy(self, p, y):
        if len(p) != len(y):
            print('They have to be the same length')
            return None
        n = len(y)
        s_true, n_true = self.entropy_of_one_division(y[p])
        s_false, n_false = self.entropy_of_one_division(y[~p])
        s = n_true * 1.0 / n * s_true + n_false * 1.0 / n * s_false
        return s

    def __find_best_split(self, col, y):
        cutoff = 0
        min_entropy = 10
        for value in set(col):
            p = col < value
            entropy = self.get_entropy(p, y)
            if entropy <= min_entropy:
                min_entropy = entropy
                cutoff = value
        return min_entropy, cutoff

    @staticmethod
    def __all_same(items):
        return all(x == items[0] for x in items)

    def predict(self, x):
        results = self.__matrix.array([0] * len(x))
        for i, c in enumerate(x):
            results[i] = self.__get_prediction(c)
        return results.reshape(-1, 1)

    def __get_prediction(self, row):
        cur_layer = self.__trees
        while cur_layer.get('cutoff'):
            if row[cur_layer['index_col']] < cur_layer['cutoff']:
                cur_layer = cur_layer['left']
            else:
                cur_layer = cur_layer['right']
        else:
            return cur_layer.get('val')


def execute(dataset_path):
    # Inject dependencies
    math = Math()
    matrix = Matrix()
    dataset = Dataset()
    decision_tree = DecisionTree(matrix, math, 100)

    data = dataset.load_dataset(dataset_path)

    # Split in Train and Test (66% - 33%)
    train, test = dataset.train_test_dataset_split(data)

    # Apply Regression
    x = matrix.df_to_array(matrix.get_x(train))
    y = matrix.series_to_ndarray(matrix.get_y(train))
    nodes = decision_tree.fit(x, y)

    # Predict
    x = matrix.df_to_array(matrix.get_x(test))
    y = matrix.series_to_ndarray(matrix.get_y(test))
    p = decision_tree.predict(x)

    return nodes, y, p


def show_metrics(nodes, y, p):
    metrics = Metrics()
    print("tree ---->", "\n", nodes, "\n")
    print("accuracy ---->", "\n", metrics.accuracy(y, p), "\n")
    print(metrics.classification_report(y, p), "\n")
    print(metrics.confusion_matrix(y, p), "\n")


def execute_diabetes():
    print("==== DIABETES")
    nodes, y, p = execute("/data/diabetes.csv")
    show_metrics(nodes, y, p)


def execute_hepatitis():
    print("==== HEPATITIS")
    nodes, y, p = execute("/data/hepatitis.csv")
    show_metrics(nodes, y, p)


def execute_iris():
    print("==== IRIS")
    nodes, y, p = execute("/data/iris.csv")
    show_metrics(nodes, y, p)


# Execute hepatitis data
execute_iris()

# Execute hepatitis data
execute_diabetes()

# Execute hepatitis data
execute_hepatitis()