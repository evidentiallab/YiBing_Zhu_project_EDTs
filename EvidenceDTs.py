from itertools import chain, combinations
import math
from pyds import MassFunction
from itertools import product
import queue
import copy

def powerset(iterable) -> frozenset:
    return map(frozenset, chain.from_iterable(combinations(iterable, r) for r in range(len(iterable) + 1)))

data = [[{frozenset({'G'}): 0.6, frozenset({'Y'}): 0.4},
  {frozenset({'G'}): 0.4, frozenset({'Y'}): 0.6},
  {frozenset({'Y'}): 0.5, frozenset({'G', 'Y'}): 0.5},
  {frozenset({'G'}): 0.3, frozenset({'Y'}): 0.7},
  {frozenset({'G'}): 0.8, frozenset({'Y'}): 0.2},
  {frozenset({'G'}): 0.1, frozenset({'Y'}): 0.9},
  {frozenset({'G'}): 0.3, frozenset({'Y'}): 0.7},
  {frozenset({'Y'}): 0.7, frozenset({'G', 'Y'}): 0.3},
  {frozenset({'Y'}): 0.2, frozenset({'G', 'Y'}): 0.8}],
 [{frozenset({'U'}): 0.9, frozenset({'O'}): 0.1},
  {frozenset({'U'}): 0.8, frozenset({'O'}): 0.1, frozenset({'L'}): 0.1},
  {frozenset({'U'}): 0.7, frozenset({'O'}): 0.3},
  {frozenset({'O'}): 0.2, frozenset({'L'}): 0.8},
  {frozenset({'U'}): 0.8, frozenset({'L', 'O', 'U'}): 0.2},
  {frozenset({'O'}): 0.3, frozenset({'L'}): 0.7},
  {frozenset({'U'}): 0.3, frozenset({'O'}): 0.7},
  {frozenset({'U'}): 0.4, frozenset({'O'}): 0.6},
  {frozenset({'O'}): 0.7, frozenset({'L'}): 0.3}]]


def set_operation(mf1_set:frozenset, mf2_set:frozenset) -> None:
    intersection = mf1_set & mf2_set
    union = mf1_set | mf2_set
    return len(intersection)/len(union)

def get_d_mf1_mf2(mf1:tuple, mf2:tuple) -> float:
    set_op_val = set_operation(mf1[0],mf2[0])
    production = mf1[1]*mf2[1]
    return set_op_val * production

def get_mf1_mf2(mf1, mf2) -> float:
    res_mf1_mf2 = 0.0
    count = 0
    for i in mf1.items():
        for j in mf2.items():
            mf1_set = tuple(i)
            mf2_set = tuple(j)
            res_mf1_mf2 =  res_mf1_mf2 + get_d_mf1_mf2(i,j)
            count = count + 1
    return res_mf1_mf2 

def distance_dbba(mf1, mf2) -> float:
    mf1_2_norm = get_mf1_mf2(mf1,mf1)
    mf2_2_norm = get_mf1_mf2(mf2,mf2)
    d_mf1_mf2 = get_mf1_mf2(mf1,mf2)
    result = math.sqrt(0.5*(mf1_2_norm+mf2_2_norm-2*d_mf1_mf2))
    return result

def similarity_attribute(mf1, mf2) -> float:
        return 1 - distance_dbba(mf1, mf2)
    
class Node:
    def __init__(self, attr = None, data = None , data_y = None):
        self.child = {}
        self.data = data 
        self.attribute = attr
        self.data_y = data_y
    
    @property
    def get_child(self):
        return self.child
    
    @property
    def get_data(self):
        return self.data
    
    def set_data(self, data):
        self.data = data
        
    def set_attr(self, attr):
        self.attribute = attr
        
    def set_data_y(self, data_y):
        self.data_y = data_y
        
class Tree:
    def __init__(self, root = None ,*,fea_label = None):
        self.root = root
        self.fea_label = fea_label
    
    def set_fea_label(self, fea_label):
        self.fea_label = fea_label
    
    def get_branch(self, mf) -> frozenset:
        max_v = 0.0
        i_v = -1
        for i , v  in mf.items():
            if v > max_v:
                max_v = v
                i_v = i
        return i_v
    
    def map_dis(self, list_d:list, list_attr:list) -> float:
        len_list = len(list_attr)
        res = 0.0
        count = 0.0
        for i in range(len_list-1):
            for j in range(i+1, len_list):
                s_2 = similarity_attribute(list_attr[i], list_attr[j])
                s_1 = similarity_attribute(list_d[i], list_d[j])
                res = res + abs(s_2 - s_1)
                count = count + 1
        return res
    
    def get_best_feature(self, data_x, data_y, data_feature) -> str:
        min_map_dis = float("inf")
        min_i = -1
        for i in range(len(data_x)):
            temp = self.map_dis(data_x[i],data_y)
            if temp < min_map_dis:
                min_map_dis = temp 
                min_i = i
        return data_feature[min_i], min_i  
      
    def set_frame_dirscrement(self,attr):
        if attr == 'ability':
            return set(['G','Y','B'])
        if attr == 'appearance' :
            return set(['G','Y'])
        if attr == 'property':
            return set(['U','O','L'])

    def fit(self, data_x, data_y, feature_name, fea_name):
        self.set_fea_label(fea_name)
        self.create_tree(n.root, data_x, data_y, feature_name)
        
    def create_tree(self, node, data_x, data_y, data_feature, depth=0):

        print("\n")
        print("depth:",depth)

        if len(data_feature) < 0:
            print("return len data_feature\n")
            node.set_data(data_x)
            node.set_data_y(data_y)
            return node 
        for i in iter(data_x):
            if len(i) <= 1:
                node.set_data(data_x)
                node.set_data_y(data_y)
                return node      
        if self.tru_lev(data_y) > 0.8:
            print("return tru_lev\n")
            node.set_data(data_x)
            node.set_data_y(data_y)
            return node
        if depth == 3:
            print("qingxiang\n")
            node.set_data(data_x)
            node.set_data_y(data_y)
            return node
        
        node.set_data(copy.deepcopy(data_x))
        node.set_data_y(copy.deepcopy(data_y))
        
        print("len of data:",len(data_x))
        best_fea, i_d = self.get_best_feature(data_x, data_y, data_feature)
        node.set_attr(i_d)
        print("best_fea:",best_fea)
        origin = self.set_frame_dirscrement(best_fea)
        pset = list(powerset(origin))
        classify_dict = {}
        classify_dict_y ={}
        data_x_now = data_x[i_d]
        
        for keys in pset:
            classify_dict[keys] = []
            for li in range(len(data_x)):
                classify_dict[keys].append([])
            classify_dict_y[keys] = []

        branch_list = []
        for i in range(len(data_x_now)):
            branch = self.get_branch(data_x_now[i])
            branch_list.append(branch)
            for li in range(len(data_x)):
                classify_dict[branch][li].append(data_x[li][i])
            classify_dict_y[branch].append(data_y[i])
#         print("classify_dict",classify_dict)
        
        #del data_feature[i_d]    
        del data_x[i_d]  
        rec = []
        for k in classify_dict.keys():
            if k not in branch_list:
                rec.append(k)

        for ik in rec:
            del classify_dict[ik]
#         print(classify_dict)
    
        for it,key in enumerate(classify_dict):
            print("key:",key)
            data_x_new = classify_dict[key]
            data_y_new = classify_dict_y[key]
            print("data_x_new:",data_x_new)
            print("len():",len(data_x_new[0]))
#             print(key)
#             print("data_y_new:", len(data_y_new))
#             print("data_x_new:",len(data_x_new[0]))
#             print("data_x_new:",data_x_new)
#             print('\n')
            node.get_child[key] = Node()
#             print("node.get_child:",node.get_child)

            node.get_child[key] = self.create_tree(node.get_child[key], data_x_new, data_y_new , data_feature, depth) 
        depth = depth + 1
        return node
           
    def tru_lev(self , data_x):
        n = len(data_x)
        l = n*n*0.5 
        res =0.0
        tru_lev_value = 0.0
        for i in range(n):
            for j in range(i+1,n):
                res = res + distance_dbba(data_x[i], data_x[j])
        tru_lev_value = 1-1/l*res
        return tru_lev_value
      
    def tree_traver(self):
        return self.traver(self.root)
    
    def traver(self, node):
        res_dict = {}
        if len(node.child) < 1:
            return node.data
        for key, val in node.child.items():
            print(key)
            res_dict[key] = self.traver(node.child[key])
        return res_dict    
    
    def get_direction(self, X, key):
        return self.get_branch(*X[key])
    
    def get_all_branch(self, mf):
        return [x for x in mf.keys()]
    
    def get_attr_i(self, attr):
        for i,k in enumerate(fea_name.keys()):
            if attr == k:
                return i
    
    def get_all_direction(self, X ,key):
        return self.get_all_branch(*X[key]) 
    
    def predict(self, X, node):
        if node.child == {}:
            node.data.append(node.data_y)
            return node.data
        branch = self.get_direction(X,node.attribute)
        return self.predict(X, node.child[branch])
   
    def tree_predict(self, X):
        r = []
        branch = self.get_all_direction(X, self.root.attribute)
        for i in iter(branch):
            r.append(self.predict(X, self.root.child[i]))
        proccess_result = self.process_leaf_node(r)
        relation_of_leaf = [] 
        leaf_data_y = []
        for j in iter(proccess_result):
            leaf_data_y.append(*j[-1:][0])
            relation_of_leaf.append(self.relation_of_two_instance(j[:-1], X, self.root.data, self.root.data_y))
        out_res = self.combine_rule(leaf_data_y, relation_of_leaf, self.fea_label['class'])
        return out_res
        
    def dis_cap(self, data_x, data_y, ik:int):
        temp = []
        record = 0.0
        for i in range(len(data_x)):
            produce = self.map_dis(data_x[i],data_y)
            if i == ik :
                record = produce
            temp.append(produce)
        dis_cap  = min(temp)/record
        return dis_cap
    
    def relation_of_two_instance(self, mf1_data, mf2_data, data_x, data_y):
        res = 0.0
        k = 0.0
        for j in range(len(data_x)):
            k = k + self.dis_cap(data_x, data_y, j)  
        for j in range(len(data_x)):
            for ji in range(len(mf1_data[j])):
                d_bba = distance_dbba(mf1_data[j][ji],mf2_data[j][ji])
                w_j = self.dis_cap(data_x, data_y, j)
                res = res+w_j*d_bba
        return 1-1/k*res 
    
    def combine_two_dict(self, mf1_r,mf2_r):
        mf1 = mf1_r
        mf2 = mf2_r
        C = {}
        if mf1 != {}:
            ap = 0.5
        else:
            ap = 1    
        for key in list(set(mf1) | set(mf2)):
            if mf1.get(key) and mf2.get(key):
                C.update({key: ap*(mf1.get(key) + mf2.get(key))})
            else:
                C.update({key: ap*(mf1.get(key) or mf2.get(key))})
        return C
    
    def combine_two_mf(self, data_x, res = {}):
        data_temp = copy.deepcopy(data_x)  # Mathis:Handling deep and shallow copies
        while data_temp:
            f = data_temp.pop(0)
            res = self.combine_two_dict(res, f)
        return res
    
    def structure_of_leaves(self, data_x):
        res = []
        ln = len(data_x)
        if ln < 2:
            return data_x
        for i in range(ln):
            res.append([self.combine_two_mf(data_x[i])])
        return res 
      
    def process_leaf_node(self, data_x, data_y=None):
        resu = []
        for x in iter(data_x):
            resu.append(self.structure_of_leaves(x))
        return resu
                       
    def change_dict(self, li:list):
        re = []
        for i in range(len(li)):
            r=[]
            for k,v in li[i].items():
                r.append(dict([(k,v)]))
            re.append(r)
        return re
    
    def cal_combine_value(self, leaf_data:list, relation_of_leaf,key=frozenset()):
        sum_all = 0.0
        table = self.change_dict(leaf_data)
        for i in product(*table):
            q=queue.Queue()
            q2=queue.Queue()
            res, m_d = 0.0, 0.0 
            t = frozenset()
            for j in iter(i):
                q.put(j)
            while True:
                if q.empty():
                    break
                out = q.get()
                q2.put(out)
                if t == frozenset():
                    t = list(out.keys())[0]
                else:t = list(out.keys())[0] & t
            if t == key:
                while True:
                    if q2.empty():
                        break
                    if res ==0.0:
                        res=res+1
                    o = q2.get()
                    res = list(o.values())[0] * res
            elif t == frozenset():
                max_len_q2 = q2.qsize()
                denominator, molecular = 0, 0
                pr = 1.0
                while True:
                    len_q2 = q2.qsize()
                    if q2.empty():
                        break
                    o = q2.get()
                    denominator = relation_of_leaf[max_len_q2-len_q2]*list(o.values())[0]+denominator
                    pr = pr * list(o.values())[0]
                    if list(o.keys())[0] == key:
                        molecular = relation_of_leaf[max_len_q2-len_q2]*list(o.values())[0]
                m_d = molecular/denominator *pr
            sum_all = sum_all + res +m_d
        return round(sum_all,2)
    
    def combine_rule(self, leaf_data:list, relation_of_leaf, baseset):
        pset = dict.fromkeys(powerset(baseset), None)
        rest_dict = {}
        for i in pset.keys():
            if i == frozenset():
                rest_dict[i] = 0
            else:rest_dict[i]=self.cal_combine_value(leaf_data, relation_of_leaf, i)
        return rest_dict
    
    def transformation(self, res_ds, baseset=None):
        baseset= self.fea_label['class']
        res_transfor = {}
        res_ds_copy = copy.deepcopy(res_ds)
        for j in iter(baseset):
            ca = 0.0
            for i, v in res_ds.items():
                if i == frozenset():
                    pass
                else:
                    fm = len(i)
                    fz = len(i&frozenset(j))
                    ca = fz/fm*v/(1-res_ds[frozenset()])+ca

            res_transfor[j]=round(ca,2)
        return res_transfor
