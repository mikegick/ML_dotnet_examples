using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;

namespace SentimentAnalysis
{
    class Program
    {
        // Location of input data
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        // Location of model
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            // Initialize context variable and load training data
            MLContext mlContext = new MLContext();
            TrainTestData splitDataView = LoadData(mlContext);

            // Build and train the model
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

            // Evaluate the model against the test set
            Evaluate(mlContext, model, splitDataView.TestSet);

            // Create single instance of test data
            // Predict "sentiment" based on test data
            // Combine test data and predictions for reporting
            // Log prediction results to console
            UseModelWithSingleItem(mlContext, model);

            // Create an array of test data
            // Predict "sentiment" based on test data
            // Combine test data and predictions for reporting
            // Log prediction results to console
            UseModelWithBatchItems(mlContext, model);
        }

        public static TrainTestData LoadData(MLContext mLContext)
        {
            // Load the data
            IDataView dataView = mLContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);

            // Split loaded dataset into train and test datasets(default is 10%, this method will use 20% for test)
            TrainTestData splitDataView = mLContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            // Returns the split train and test datasets
            return splitDataView;
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainSet)
        {
            // Convert "Sentiment Text" column into numeric key types used by ML algorithm and add as a new dataset column
            // Append a classification algorithm for categorizing rows as positive or negative (binary classification)
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
            // Info on SdcaLogisticRegression training algorithm: https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.sdcalogisticregressionbinarytrainer

            // Train the model
            Console.WriteLine("========== Create and Train the Model ==========");
            var model = estimator.Fit(trainSet);
            Console.WriteLine("========== Training Complete ==========\n");

            // Return the trained model
            return model;
        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView testSet)
        {
            // Transform the test set in order to make predictions
            Console.WriteLine("========== Evaluating Model Accuracy with Test Data ==========");
            IDataView predictions = model.Transform(testSet);

            // Compare predicted values with actual labels to assess model's performance
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            // Display metrics
            // Accuracy proportion of correct predictions in the test set
            // AUC is model confidence in classifying positive/negative classes. The closer to 1, the better
            // F1 Score is measure of balance between precision and recall. The closer to 1, the better
            Console.WriteLine();
            Console.WriteLine("Model Quality Metrics Evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Area Under Roc Curve: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("========== End of Model Evaluation ==========");
        }

        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            // Perform prediction on a single instance of data
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            // PredictionEngine is a Convenience API. To learn more, visit https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.predictionengine-2

            // Create a sample data point to test against
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This was a very bad steak"
            };

            // Make a prediction on a single data point (the sample data point)
            var resultPrediction = predictionFunction.Predict(sampleStatement);

            // Log the sentiment text as well as the prediction
            Console.WriteLine();
            Console.WriteLine("========== Prediction Test of model with a single sample and test dataset ==========");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText}" +
                $"| Prediction: " +
                $"{(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")}" +
                $" | Probability: {resultPrediction.Probability} ");

            Console.WriteLine("========== End of Predictions ==========");
            Console.WriteLine();
        }

        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            // Create multiple sample data points for testing the model on a set
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This was a horrible meal"
                },
                new SentimentData
                {
                    SentimentText = "I love this spaghetti."
                }
            };

            // Use the model to predict the comment data sentiment using the Transform() method
            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);
            IDataView predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            // Log the sentiment text as well as the prediction for each sample data point
            Console.WriteLine();
            Console.WriteLine("========== Prediction Test of loaded model with multiple samples ==========");

            foreach (var prediction in predictedResults) {
                Console.WriteLine();
                Console.WriteLine($"Sentiment: {prediction.SentimentText}" +
                    $"| Prediction: " +
                    $"{(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")}" +
                    $" | Probability: {prediction.Probability} ");
            }

            Console.WriteLine("========== End of Predictions ==========");
            Console.WriteLine();
        }
    }
}
