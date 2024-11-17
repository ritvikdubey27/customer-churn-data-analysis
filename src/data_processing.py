from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
import os

def create_spark_session():
    """Create and return a Spark session"""
    return SparkSession.builder \
        .appName("CustomerChurnAnalysis") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

def load_data(spark, file_path):
    """Load data from CSV file"""
    try:
        return spark.read.csv(file_path, header=True, inferSchema=True)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def clean_data(df):
    """Clean the dataframe by handling missing values and converting data types"""
    try:
        # Convert TotalCharges to DoubleType and handle empty strings
        df = df.withColumn("TotalCharges", 
            when(col("TotalCharges").isNull() | (col("TotalCharges") == " "), 0)
            .otherwise(col("TotalCharges").cast(DoubleType())))
        
        # Convert binary categorical variables to numeric
        binary_columns = ["SeniorCitizen", "Partner", "Dependents", 
                         "PhoneService", "PaperlessBilling", "Churn"]
        
        for column in binary_columns:
            df = df.withColumn(column, 
                when(col(column) == "Yes", 1)
                .otherwise(0))
        
        return df
    except Exception as e:
        print(f"Error cleaning data: {str(e)}")
        raise

def prepare_features(df):
    """Prepare features for machine learning"""
    try:
        # Identify categorical and numeric columns
        categorical_cols = [field for (field, dataType) in df.dtypes 
                          if dataType == "string" and field != "customerID"]
        
        numeric_cols = [field for (field, dataType) in df.dtypes 
                       if ((dataType == "double") | (dataType == "int")) 
                       and (field != "Churn")]

        # Create pipeline stages
        stages = []

        # String Indexing for categorical variables
        for categoricalCol in categorical_cols:
            stringIndexer = StringIndexer(
                inputCol=categoricalCol, 
                outputCol=categoricalCol + "Index",
                handleInvalid="keep"
            )
            encoder = OneHotEncoder(
                inputCols=[stringIndexer.getOutputCol()],
                outputCols=[categoricalCol + "classVec"]
            )
            stages += [stringIndexer, encoder]

        # Assemble all features into a single vector
        assemblerInputs = [c + "classVec" for c in categorical_cols] + numeric_cols
        assembler = VectorAssembler(
            inputCols=assemblerInputs,
            outputCol="features",
            handleInvalid="keep"
        )
        stages += [assembler]

        # Create and fit the pipeline
        pipeline = Pipeline(stages=stages)
        pipelineModel = pipeline.fit(df)
        df = pipelineModel.transform(df)

        return df, pipelineModel
    
    except Exception as e:
        print(f"Error preparing features: {str(e)}")
        raise

def main():
    try:
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Create necessary directories
        os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
        os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
        
        # Initialize Spark session
        spark = create_spark_session()
        
        # Load data
        file_path = os.path.join(project_root, "data", "customer_churn.csv")
        print(f"Loading data from: {file_path}")
        df = load_data(spark, file_path)
        
        # Print initial dataset info
        print("\nInitial Dataset Info:")
        print(f"Number of records: {df.count()}")
        print(f"Number of features: {len(df.columns)}")
        
        # Clean data
        print("\nCleaning data...")
        df_cleaned = clean_data(df)
        
        # Prepare features
        print("\nPreparing features...")
        df_prepared, pipeline_model = prepare_features(df_cleaned)
        
        # Select relevant columns for the final dataset
        final_df = df_prepared.select("customerID", "features", "Churn")
        
        # Show sample of the final dataset
        print("\nSample of the final dataset:")
        final_df.show(5, truncate=False)
        
        # Save the processed data
        output_path = os.path.join(project_root, "data", "processed_churn_data.parquet")
        print(f"\nSaving processed data to: {output_path}")
        final_df.write.parquet(output_path, mode="overwrite")
        
        # Save the pipeline model
        model_path = os.path.join(project_root, "models", "feature_pipeline_model")
        print(f"Saving pipeline model to: {model_path}")
        pipeline_model.save(model_path)
        
        print("\nProcessing completed successfully!")
        
        spark.stop()
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        if 'spark' in locals():
            spark.stop()
        raise

if __name__ == "__main__":
    main()