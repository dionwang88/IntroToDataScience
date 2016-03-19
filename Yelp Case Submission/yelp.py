df_review = sqlContext.read.json("../reviews.json")
yelp_users = sqlContext.read.json("../yelp_user.json")
yelp_business = sqlContext.read.json('../yelp_business.json')

sqlContext.registerDataFrameAsTable(df_review, "reviews_json")
sqlContext.registerDataFrameAsTable(yelp_users, "yelp_users")
sqlContext.registerDataFrameAsTable(yelp_business, "yelp_business")

df_users = sqlContext.sql('select distinct user_id, count(review_id) as count '\
                            'from reviews_json group by user_id having count > 80')

user_count = sqlContext.sql('select distinct user_id, count(stars) as count '\
                            'from reviews_json group by user_id order by count desc')


df_users.registerTempTable("user")
df_users1 = sqlContext.sql('select u.user_id, u.count, y.average_stars, y.review_count, y.name'\
                           ' from user as u, yelp_users as y'\
                           ' where u.user_id == y.user_id')
df_users1.registerTempTable("user")

df2 = sqlContext.sql('select r.user_id, r.business_id,r.stars from reviews_json as r, user as u where r.user_id = u.user_id')
# df2.count()
# df2.cache
df_business = sqlContext.sql('select distinct business_id from reviews_json, user where reviews_json.user_id = user.user_id')
# df_business.count()
df_business.registerTempTable("business")



# Unstack and Transpose in Pandas
pd_raw = df2.toPandas()
pd1 = pd_raw.drop_duplicates(['user_id','business_id'])
pd1.count()
pd2 = pd1.pivot('user_id','business_id','stars')
pd3 = pd2.replace('NaN',0.1)
pd3.to_csv("../yelp.csv")
# transpose :
pd3_trans = pd3.transpose()
pd3_trans.to_csv("../yelp_trans.csv")

def removeHeader(itr_index, itr):
    return iter(list(itr)[1:]) if itr_index == 0 else itr

def removeColumns(line):
    items = [(n.strip()) for n in line.split(",")]
    items = [float(n) for n in items[1:]]
    return np.array(items)

def readRankMatrix():
    import numpy as np
    lines = sc.textFile('../yelp_trans.csv')
    rawData = lines.mapPartitionsWithIndex(removeHeader)
    mydata = rawData.map(removeColumns).cache()
    return mydata


from pyspark.mllib.stat import Statistics
from pandas import Series
import pandas as pd
import numpy as np
import math

mydata = readRankMatrix()
corr = Statistics.corr(mydata)

# set up the columns names and add a new names called user_id
lines2 = sc.textFile('../yelp.csv')
names = lines2.map(lambda line:line.split(",")).map(lambda a:a[0]).collect()[1:]

s = Series([str for str in names])
pddata = pd.DataFrame(corr, columns=s)
pddata['user_id'] = names
df_corr = sqlContext.createDataFrame(pddata)
# df_corr.cache()
df_corr.registerTempTable("corr")

def getTopReviewUsers(n):
    # n: the nth highest user
    ord_user = sqlContext.sql("select user_id, count(review_id) as count from reviews_json group by user_id order by count desc")
    user_list = ord_user.collect()
    # get the user_id who has the most number of reviews
    user_id = user_list[n-1][0]
    return user_id


def getTopNeighbors(uid, n):
    aa_col = df_corr.orderBy(uid, ascending=0)\
                    .select(df_corr.user_id,uid).collect()
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

# get the business_id which has most reviews
def getTopBusiness(n):
    # n: the top nth highest business with reviews
    business = sqlContext.sql('select business_id,count(review_id) \
                                from review group by business_id \
                                order by count(review_id) desc')
    bid = business.collect()[n-1][0]
    return bid
# bid = getTopBusiness(1)

def getTopN_User_Business_Rank(uid, topN_neighbor):
    # uid: the user_id that we want to predict
    # topN_neighbor: row[0]-user_id, row[1]-correlation
    # return dict: key is user_id+business_id, value is star of this user to this business
    neighbor_names = "'" + uid + "'"
    for row in topN_neighbor:
        neighbor_names = neighbor_names + ','+ "'" + row[0] + "'"
    sql_str = 'select user_id, business_id, stars '\
              'from reviews_json where user_id in (' + neighbor_names + ')'
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

uid = getTopReviewUsers(7)
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

# Export result to csv file
k = []
RMSE = []
Accuracy = []
for t in result_list:
  k.append(t.k)
  RMSE.append(t.RMSE)
  Accuracy.append(t.accuracy)

d = {'K':k, 'RMSE':RMSE,'Accuracy':Accuracy}
pd_result = pd.DataFrame(d)


# To do prediction for user using rank-based collaborative filtering
k = 45
u_bus = sqlContext.sql('select distinct business_id from yelp_business').collect()
topN_neighbor = getTopNeighbors(uid, k)
topN_std = {}
for row in topN_neighbor:
    topN_std[row[0]] = getStdRankofUser(row[0])

# add the target user's standard deviation to the topN users' std dict
topN_std[uid] = getStdRankofUser(uid)
u_bus_dict = getTopN_User_Business_Rank(uid, topN_neighbor)

# get the business list of the target user_id
rank_pred_list = []
for row in u_bus:
    bid = row[0]
    rank_pred = predictRank(uid, bid, avg_u, topN_neighbor, topN_std, u_bus_dict)
    rank_pred_list.append((bid, rank_pred))

def insertion_sort(lst):
    if len(lst) == 1:
        return

    for i in xrange(1, len(lst)):
        temp = lst[i]
        j = i - 1
        while j >= 0 and temp[1] < lst[j][1]:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = temp

# for the seventh user prediction:
# user_id = fczQCSmaWF78toLEmb0Zsw, rank_predict = 5.52059073685, business _id = a56tFBfDCjx_894OWhQgKA
uid = 'fczQCSmaWF78toLEmb0Zsw'
bid = 'IWwt0vm8lYbVoyGSPd2Z9A'
rank_predict = 5.52059073685
recommend_business = sqlContext.sql("select business_id, name, state, city, stars, review_count "\
                                    "from yelp_business where business_id = '" + bid + "'").collect()
recommend_user = sqlContext.sql("select user_id, name from yelp_users "\
                                "where user_id = '" + uid + "'").collect()
