import numpy as np

def gini(y):
    """
    y:data_label
    """
    l_y = len(y)
    labels = np.unique(y)
    gini = 0
    for label in labels:
        ans = y[np.where(y[:]==label)].size/l_y # the probability of occurrence of the label
        gini -= ans*ans
    return gini

class Node():
    def __init__(self, feature = None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature 
        self.threshold = threshold
        self.left = left
        self.right = right 
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None   
    
    def get_left_node(self):
        return self.left
    
    def get_right_node(self):
        return self.right
    
class MDSTree():
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self._root = None 
        self.record = []
       
        
    def predict(self, X):
        return np.array([self._traverse_tree(x, self._root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
        
    def _split_data(self, data_x, data_y, fea_axis, fea_value):
        if  isinstance(fea_value,float):
            equal_Idx = np.where(data_x[:,fea_axis]>=fea_value)
            nequal_Idx = np.where(data_x[:,fea_axis]<fea_value)
        else:
            equal_Idx = np.where(data_x[:,fea_axis]==fea_value) 
            nequal_Idx = np.where(data_x[:,fea_axis]!=fea_value)
        return data_x[equal_Idx],data_y[equal_Idx],data_x[nequal_Idx],data_y[nequal_Idx]


    def _get_best_feature(self, data_x, data_y):
        m,n = data_x.shape
        best_fea = -1
        best_fea_val = -1
        min_fea_gini = np.inf
    
        for i in range(n):
            feas = np.unique(data_x[:,i]) 
            for j in feas:
                equal_data_x,equal_data_y,nequal_data_x,nequal_data_y = self._split_data(data_x,data_y,i,j)
                fea_gini = 0.0
            
                fea_gini = len(equal_data_y)/m*gini(equal_data_y)+len(nequal_data_y)/m*gini(nequal_data_y)
                if fea_gini<min_fea_gini:
                    min_fea_gini = fea_gini
                    best_fea = i
                    best_fea_val = j
        return best_fea,best_fea_val
    
    def fit(self, X, y, features_names):
        self.features_names = features_names
        self._root = self._create_tree(X, y, features_names)

    def _create_tree(self, X, y ,fea_label, depth=0):
        labels = np.unique(y)
        if len(labels)==1:
            return Node(value = y[0] )
        if X.shape[1]==0:
            best_fea, best_fea_num = 0,0
            for label in labels:
                num = y[np.where(y==label)].size
                if num > best_fea_num:
                    best_fea = label
                    best_Fea_num = num
            return Node(value = best_fea)
        best_fea, best_fea_val =self._get_best_feature(X,y)
        best_fea_label = fea_label[best_fea]
        print(u"此时最优索引为："+str(best_fea_label))
        self.record.append(str(best_fea_label))
        equal_data_x,equal_data_y,nequal_data_x,nequal_data_y = self._split_data(X,y,best_fea,best_fea_val)
        
        # equal_data_x = np.delete(equal_data_x,best_fea,1)
        # nequal_data_x = np.delete(nequal_data_x,best_fea,1)    
        # fea_label = np.delete(fea_label,best_fea,0)
        left = self._create_tree(nequal_data_x,nequal_data_y,fea_label, depth+1)
        right = self._create_tree(equal_data_x,equal_data_y,fea_label, depth+1)
        return Node(best_fea, best_fea_val, left, right)
    
    @property
    def get_root(self):
        return self._root
    
    def get_level_ord(self):
        self._level =self.row_order(self.get_root())
        return self._level
    
    def row_order(self, root):
        res, queue = [], []
        if root == None:
            return []
        queue.append(root)
        while queue:
            size = len(queue)
            level = []
            for i in range(size):
                node = queue.pop(0)
                if node.feature != None:
                    str_temp = str(self.features_names[node.feature])+str(" ")+str(node.threshold)
                    level.append(str_temp)
                else:
                    level.append(node.value)
                    
                if node.left != None:
                    queue.append(node.left)
                if node.right != None:
                    queue.append(node.right)
            res.append(level)
        return res
        
    def get_tree_dict(self):
         return self.tree_to_dict(self.get_root())
        
    def tree_to_dict(self, node, tree_dict={}, depth = 0, loc = 'L'):
        if node.get_left_node() == None and node.get_right_node() == None:
            node_name = str(depth)+loc+str(" ")+"leaf node"
            tree_dict[node_name] = node.value
            return tree_dict[node_name]
        node_name = str(depth)+loc+str(" ")+str(self.features_names[node.feature])+str(node.threshold)
        tree_dict[node_name] = {}
        if node.left != None:
            self.tree_to_dict(node.get_left_node(), tree_dict[node_name], depth+1, loc='L')
        if node.get_right_node() != None:
            self.tree_to_dict(node.get_right_node(), tree_dict[node_name], depth+1, loc='R')
        return tree_dict
    

# if __name__ =="__main__":
#     from sklearn import datasets
#     from sklearn.model_selection import train_test_split
#     def accuracy(y_true, y_pred):
#         accuracy = np.sum(y_true == y_pred) / len(y_true)
#         return accuracy
#     data = datasets.load_iris()
#     X = data.data
#     y = data.target
#     X_name = data.feature_names
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=666)

#     ct = MDSTree(max_depth=10)
#     ct.fit(X_train, y_train, X_name)

#     y_pred = ct.predict(X_test)
#     acc = accuracy(y_test, y_pred)
#     print ("Accuracy:", acc)    
#     print(ct.get_root.right.left.left.feature)