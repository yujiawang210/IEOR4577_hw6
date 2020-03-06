import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pre_processing.pre_processing import PreProcessor

my_pre_processor = PreProcessor()

args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
datasource0 = glueContext.create_dynamic_frame.from_catalog(database = "hw6", table_name = "train_csv", transformation_ctx = "datasource0")

applymapping1 = ApplyMapping.apply(frame = datasource0, mappings = [("sentiment", "long", "sentiment", "long"), ("tweet", "string", "tweet", "string")], transformation_ctx = "applymapping1")

def map_function(dynamicRecord):
    tweet=dynamicRecord["tweet"]
    features=my_pre_processor.pre_process_text(tweet)
    dynamicRecord["features"]=features
    return dynamicRecord

mapping1= Map.apply(frame = applymapping1, f = map_function, transformation_ctx = "mapping1")

datasink2 = glueContext.write_dynamic_frame.from_options(frame = mapping1, connection_type = "s3", connection_options = {"path": "s3://ieor4577-hw6/train"}, format = "json", transformation_ctx = "datasink2")
job.commit()
