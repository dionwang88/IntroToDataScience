df_review = sqlContext.sql('select * from reviews_json')
from pyspark.mllib.stat import Statistics
from pandas import Series
import pandas as pd
import numpy as np
import math

df_users = sqlContext.sql('select distinct user_id from reviews_json group by user_id having count(review_id) > 80').cache()
df_users.registerTempTable("user")
df2 = sqlContext.sql('select r.user_id, r.business_id,r.stars from reviews_json as r, user as u where r.user_id = u.user_id').cache()
df_business = sqlContext.sql('select distinct business_id from reviews_json, user where reviews_json.user_id = user.user_id')
df_business.registerTempTable("business")

review_count = sqlContext.sql('select distinct user_id,count(review_id) as count from reviews_json group by user_id having count(review_id) > 80 order by count desc')

# print review_count.collect()[0]

review_count = sqlContext.sql('select distinct user_id,count(review_id) as count from reviews_json group by user_id having count(review_id) > 80 order by count desc')

# print review_count.collect()[0]

pd_raw = df2.toPandas()
pd1 = pd_raw.drop_duplicates(['user_id','business_id'])

pd2 = pd1.pivot('user_id','business_id','stars')
pd3 = pd2.replace('NaN',0.1)
pd3_trans = pd3.transpose()

df_pd3 = sqlContext.createDataFrame(pd3)
df_pd3_trans = sqlContext.createDataFrame(pd3_trans)

def createMatrix(r):
  items = [float(t) for t in r]
  return np.array(items)

mydata = df_pd3_trans.map(createMatrix)
corr = Statistics.corr(mydata)

index_names = pd3.index.values
s = Series([str for str in index_names])
pddata = pd.DataFrame(corr, columns=s)
pddata['user_id'] = index_names
df_corr = sqlContext.createDataFrame(pddata)
df_corr.registerTempTable("corr")

def getTopReviewUsers(n):
    # n: the nth highest user
    ord_user = sqlContext.sql("select user_id, count(review_id) from reviews_json group by user_id order by count(review_id) desc")
    user_list = ord_user.collect()
    # get the user_id who has the most number of reviews
    user_id = user_list[n-1][0]
    return user_id

  # get the business_id which has most reviews
def getTopBusiness(n):
    # n: the top nth highest business with reviews
    business = sqlContext.sql('select business_id,count(review_id) \
                                from reviews_json group by business_id \
                                order by count(review_id) desc')
    bid = business.collect()[n-1][0]
    return bid
bid = getTopBusiness(1)
print bid

def getTopNeighbors(uid, n):
    aa_col = df_corr.orderBy(uid, ascending=0).select(df_corr.user_id,uid).collect()
    top_neighbor = aa_col[1:n+1] # Because the first is itself
    return top_neighbor

    def getStdRankofBusiness(bid):
    std_rank_business = np.std(df_review.filter(df_review.business_id == bid)\
                                .select(df_review.stars)\
                                .map(lambda a:a[0])\
                                .collect())
    return std_rank_business


def getStdRankofUser(uid):
    std_rank_user = np.std(df_review.filter(df_review.user_id == uid)\
                                .select(df_review.stars)\
                                .map(lambda a:a[0])\
                                .collect())
    return std_rank_user

def getRankofUserofBusiness(uid,bid):
    rank = df_review.filter(df_review.business_id == bid)\
                    .filter(df_review.user_id == uid)
    if rank.count() > 0:
        return rank.select(df_review.stars).collect()[0][0]
    else:
        return 0

def getTopN_User_Business_Rank(uid, topN_neighbor):
    # uid: the user_id that we want to predict
    # topN_neighbor: row[0]-user_id, row[1]-correlation
    # return dict: key is user_id+business_id, value is star of this user to this business
    neighbor_names = "'" + uid + "'"
    for row in topN_neighbor:
        neighbor_names = neighbor_names + ','+ "'" + row[0] + "'"
    sql_str = 'select user_id, business_id, stars from reviews_json where user_id in (' + neighbor_names + ')'

    df_u_b = sqlContext.sql(sql_str).collect()
    u_bus_dict = {}
    for row in df_u_b:
        u_bus_dict[row[0]+row[1]] = row[2]
    return u_bus_dict

def predictRank(uid, bid, avg_u, topN_neighbor, topN_std, u_bus_dict):
    # uid : the user id
    # bid : business id
    # avg_u: the dict of average rank of users
    # topN_neighbor: row[0]-user_id, row[1]-correlation
    # topN_std: the dict of standard deviation of top N users
    # u_bus_dict: user_business dictionary, key:user_id+bisiness_id, value:rank

    # get average rank of this user_id
    avg_rank_user = avg_u[uid]
    std_rank_user = topN_std[uid]

    total_weights = 0
    total_delta = 0
    rank_ai = 0

    for row in topN_neighbor:
        uid_u = row[0]
        weight_u = row[1]
        key = uid_u + bid
        u_rank = 0

        if key in u_bus_dict:
            u_rank = u_bus_dict[key]
            ru_mean = avg_u[uid_u]
            std_u = topN_std[uid_u]
            total_delta = total_delta + (u_rank - ru_mean)*weight_u/std_u
            total_weights = total_weights + weight_u
    if total_delta == 0 or total_weights == 0:
        return avg_rank_user
    else:
        rank_ai = avg_rank_user + std_rank_user * total_delta / total_weights
        return rank_ai

# get the dict of average ranking for business and users
avg_rank_b = df_review.groupBy('business_id')\
                    .agg({'stars':'mean'})\
                    .collect()
avg_b = {}
for row in avg_rank_b:
    avg_b[row[0]] = row[1]

avg_rank_u = df_review.groupBy('user_id')\
                    .agg({'stars':'mean'})\
                    .collect()
avg_u = {}
for row in avg_rank_u:
    avg_u[row[0]] = row[1]

uid = getTopReviewUsers(7) # u'fczQCSmaWF78toLEmb0Zsw'
u_bus = df_review.filter(df_review.user_id == uid)\
                .select(df_review.business_id,df_review.stars)\
                .collect()

class Result:
    def __init__(self, k, RMSE,accuracy):
        self.k = k
        self.RMSE = RMSE
        self.accuracy = accuracy

# 1. get the user-business list
# 2. remove the businesses that rank is zero
# 3. loop the non-zero user-business list to calculate the rank
# 4. calcuate the RMSE, Accuracy
result_list = []

for k in range(5,50,5):
    # get all related users (including target user and near neighbors) business stars
    # Calculate the topN user's std
    topN_neighbor = getTopNeighbors(uid, k)
    topN_std = {}
    for row in topN_neighbor:
        topN_std[row[0]] = getStdRankofUser(row[0])

    # add the target user's standard deviation to the topN users' std dict
    topN_std[uid] = getStdRankofUser(uid)
    u_bus_dict = getTopN_User_Business_Rank(uid, topN_neighbor)

    # get the business list of the target user_id

    rank_dict_true = {}
    rank_dict_pred = {}
    for row in u_bus:
        bid = row[0]
        rank = row[1]
        rank_pred = predictRank(uid, bid, avg_u, topN_neighbor, topN_std, u_bus_dict)
        rank_dict_true[bid] = rank
        rank_dict_pred[bid] = rank_pred

    # calcuate the Accuracy
    lst = []
    rmse_lst = []
    for item in rank_dict_pred:
        pred = rank_dict_pred[item]
        true_value = rank_dict_true[item]
        acu = abs(pred-true_value)/true_value
        rse = (pred-true_value)**2
        lst.append(acu)
        rmse_lst.append(rse)

    accuracy = 1 - sum(lst)/len(lst)    # 82.6% uid='fczQCSmaWF78toLEmb0Zsw' k=30

    RMSE = math.sqrt(sum(rmse_lst)/len(rmse_lst)) # RMSE = 0.778
    r = Result(k,RMSE,accuracy)
    result_list.append(r)

for t in result_list:
  print t.k, t.RMSE, t.accuracy
