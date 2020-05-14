from pyspark import SparkContext
from pyspark.sql import SparkSession
# from pyspark.sql.functions import *
import pyspark.sql.functions as F
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import col
from pyspark.sql.functions import round
from pyspark.sql.types import StringType
from pyspark import SQLContext
from itertools import islice

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

from mmlspark.lightgbm import LightGBMClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import matplotlib. pyplot as plt
#creating the context
sqlContext = SQLContext(sc)

class crash_process():
    def __init__(self, feats, categorical_columns):
        self.feats = feats
        self.categorical_columns = categorical_columns
        
    # One-hot encoding for categorical columns with get_dummies, get binary attributes
    def one_hot_encoder(self, df):
        # The index of string vlaues multiple columns
        indexers = [
            StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
            for c in self.categorical_columns
        ]

        # The encode of indexed vlaues multiple columns
        encoders = [OneHotEncoder(dropLast=False,inputCol=indexer.getOutputCol(),
                    outputCol="{0}_encoded".format(indexer.getOutputCol())) 
            for indexer in indexers
        ]

        # Vectorizing encoded values
        assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders],outputCol="features")

        pipeline = Pipeline(stages=indexers + encoders+[assembler])
        model=pipeline.fit(df)
        transformed = model.transform(df)
        return transformed
    
    def preprocess(self, path, fraction = 0.8, seed = 110):
        # df1 = spark.read.csv("s3://vacarcrash/Virginia_Crashes.csv")
        #reading the virginia car crash csv file and store it in an RDD
        rdd= sc.textFile(path).map(lambda line: line.split(","))
        #removing the first row as it contains the header
        rdd = rdd.mapPartitionsWithIndex(
        lambda idx, it: islice(it, 1, None) if idx == 0 else it
        )

        #converting the RDD into a dataframe
        df = rdd.toDF(['collision', 'weather', 'vehicles', 'light', 'surface', 'location',
               'relation', 'hour', 'second', 'year', 'severity', 'ped_injury',
               'ped_fatality', 'total_injury', 'total_fatality', 'work_zone',
               'school_zone', 'route', 'district', 'minus', 'index'])
        #print the dataframe
        df = df.withColumn('hour', (round(df['hour']).astype('int')))
        
        # select features to be used
        df = df[self.feats]
        df = df[~(df['severity']== 5)]
        
        # convert severity to fatality
        exprs = [F.when(F.col('severity') == 1,1).otherwise(0).alias('fatality')]
        df = df.select(exprs+df.columns)
        df = self.one_hot_encoder(df)
        df.printSchema()
        
        #sample 20% dataset as test dataset
        (train_df, test_df) = df.randomSplit([0.8, 0.2])
        return [train_df, test_df]


## run preprocess
feats = ['weather', 'light', 'surface','location','hour', 'severity', 'school_zone', 'route','index']
categorical_column = ['weather', 'light', 'surface','school_zone', 'route', 'location']
path = "s3://vacarcrash/"

cr = crash_process(feats, categorical_column)
train_df, test_df = cr.preprocess(path + 'Virginia_Crashes.csv')


def lgbm_vacarcrash(feature, label, learningRate=0.1,numIterations=1000,earlyStoppingRound=10):
    lgb_estimator = LightGBMClassifier(learningRate=learningRate, 
                                       numIterations=numIterations,
                                       earlyStoppingRound=earlyStoppingRound,
                                       labelCol = label)
    
    paramGrid = ParamGridBuilder().addGrid(lgb_estimator.numLeaves, [30, 50]).build()
    eval = BinaryClassificationEvaluator(labelCol = label, metricName="areaUnderROC")
    crossval = CrossValidator(estimator=lgb_estimator,
                              estimatorParamMaps=paramGrid, 
                              evaluator=eval, 
                              numFolds=3)
    cvModel  = crossval.fit(train_df[[feature, label]])
    return cvModel, paramGrid


(cvModel, paramGrid) = lgbm_vacarcrash("features", "fatality")
print(list(zip(cvModel.avgMetrics, paramGrid)))

class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)

    def _to_list(self, rdd):
        points = []
        for row in rdd.collect():
            points += [(float(row._1()), float(row._2()))]
        return points

    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)

bestModel = cvModel.bestModel
predictions = bestModel.transform(test_df)

# Returns as a list (false positive rate, true positive rate)
preds = predictions.select('fatality','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['fatality'])))
points = CurveMetrics(preds).get_curve('roc')

x_val = [x[0] for x in points]
y_val = [x[1] for x in points]
plt.title("ROC AUC")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.plot(x_val, y_val)
plt.show()

import boto3
import json
s3 = boto3.resource('s3')
bucket_name = 'vacarcrash'
file_name = 'points'
object = s3.Object(bucket_name, file_name)
object.put(Body=(bytes(json.dumps(points, indent=2).encode('UTF-8'))))
