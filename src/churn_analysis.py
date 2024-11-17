from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, round, avg, when, desc
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def create_spark_session():
    """Create and return a Spark session"""
    return SparkSession.builder \
        .appName("ChurnAnalysis") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

def load_processed_data(spark, file_path):
    """Load the processed parquet data"""
    try:
        return spark.read.parquet(file_path)
    except Exception as e:
        print(f"Error loading processed data: {str(e)}")
        raise

def perform_exploratory_analysis(df):
    """Perform exploratory data analysis"""
    try:
        # Calculate churn rate
        churn_rate = df.filter(col("Churn") == 1).count() / df.count() * 100
        print(f"\nOverall Churn Rate: {churn_rate:.2f}%")

        # Convert Spark DataFrame to Pandas for visualization
        pandas_df = df.toPandas()
        
        # Create visualizations directory if it doesn't exist
        os.makedirs("visualizations", exist_ok=True)

        # Create churn distribution plot
        plt.figure(figsize=(8, 6))
        sns.countplot(data=pandas_df, x='Churn')
        plt.title('Distribution of Churn')
        plt.savefig('visualizations/churn_distribution.png')
        plt.close()

        return churn_rate
    except Exception as e:
        print(f"Error in exploratory analysis: {str(e)}")
        raise

def train_test_split(df):
    """Split the data into training and testing sets"""
    try:
        # Split the data 70-30
        train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)
        return train_data, test_data
    except Exception as e:
        print(f"Error in train-test split: {str(e)}")
        raise

def train_logistic_regression(train_data, test_data):
    """Train and evaluate a logistic regression model"""
    try:
        # Initialize Logistic Regression model
        lr = LogisticRegression(featuresCol="features", labelCol="Churn")

        # Create ParamGrid for Cross Validation
        paramGrid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.01, 0.1, 0.3]) \
            .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
            .build()

        # Create CrossValidator
        crossval = CrossValidator(estimator=lr,
                                estimatorParamMaps=paramGrid,
                                evaluator=BinaryClassificationEvaluator(labelCol="Churn"),
                                numFolds=3)

        # Fit the model
        cv_model = crossval.fit(train_data)
        
        # Make predictions
        predictions = cv_model.transform(test_data)
        
        # Evaluate the model
        evaluator = BinaryClassificationEvaluator(labelCol="Churn")
        auc = evaluator.evaluate(predictions)
        
        print(f"\nLogistic Regression Model AUC: {auc:.4f}")
        
        return cv_model, predictions, auc
    except Exception as e:
        print(f"Error in logistic regression training: {str(e)}")
        raise

def train_random_forest(train_data, test_data):
    """Train and evaluate a random forest model"""
    try:
        # Initialize Random Forest model
        rf = RandomForestClassifier(featuresCol="features", labelCol="Churn")

        # Create ParamGrid for Cross Validation
        paramGrid = ParamGridBuilder() \
            .addGrid(rf.numTrees, [10, 20, 30]) \
            .addGrid(rf.maxDepth, [5, 10]) \
            .build()

        # Create CrossValidator
        crossval = CrossValidator(estimator=rf,
                                estimatorParamMaps=paramGrid,
                                evaluator=BinaryClassificationEvaluator(labelCol="Churn"),
                                numFolds=3)

        # Fit the model
        cv_model = crossval.fit(train_data)
        
        # Make predictions
        predictions = cv_model.transform(test_data)
        
        # Evaluate the model
        evaluator = BinaryClassificationEvaluator(labelCol="Churn")
        auc = evaluator.evaluate(predictions)
        
        print(f"Random Forest Model AUC: {auc:.4f}")
        
        return cv_model, predictions, auc
    except Exception as e:
        print(f"Error in random forest training: {str(e)}")
        raise

def save_results(lr_predictions, rf_predictions, project_root):
    """Save the prediction results"""
    try:
        # Create results directory
        results_path = os.path.join(project_root, "results")
        os.makedirs(results_path, exist_ok=True)
        
        # Save predictions
        lr_predictions.write.parquet(os.path.join(results_path, "lr_predictions.parquet"), mode="overwrite")
        rf_predictions.write.parquet(os.path.join(results_path, "rf_predictions.parquet"), mode="overwrite")
        
        print("\nPrediction results saved successfully!")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        raise

def main():
    try:
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Initialize Spark session
        spark = create_spark_session()
        
        # Load processed data
        processed_data_path = os.path.join(project_root, "data", "processed_churn_data.parquet")
        df = load_processed_data(spark, processed_data_path)
        
        print("Performing Exploratory Analysis...")
        churn_rate = perform_exploratory_analysis(df)
        
        print("\nSplitting data into training and testing sets...")
        train_data, test_data = train_test_split(df)
        
        print("\nTraining Logistic Regression Model...")
        lr_model, lr_predictions, lr_auc = train_logistic_regression(train_data, test_data)
        
        print("\nTraining Random Forest Model...")
        rf_model, rf_predictions, rf_auc = train_random_forest(train_data, test_data)
        
        # Save results
        save_results(lr_predictions, rf_predictions, project_root)
        
        # Print final results
        print("\nFinal Results:")
        print(f"Overall Churn Rate: {churn_rate:.2f}%")
        print(f"Logistic Regression AUC: {lr_auc:.4f}")
        print(f"Random Forest AUC: {rf_auc:.4f}")
        
        spark.stop()
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        if 'spark' in locals():
            spark.stop()
        raise

if __name__ == "__main__":
    main()