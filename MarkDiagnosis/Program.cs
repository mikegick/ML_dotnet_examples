using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;

namespace MarkDiagnosis
{
    class Program
    {
        // Location of input data
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "APS_Pages.txt");
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
        }

        public static TrainTestData LoadData(MLContext mLContext)
        {
            // Load the data
            IDataView dataView = mLContext.Data.LoadFromTextFile<PageData>(_dataPath, hasHeader: false);

            // Split loaded dataset into train and test datasets(default is 10%, this method will use 20% for test)
            TrainTestData splitDataView = mLContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            // Returns the split train and test datasets
            return splitDataView;
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainSet)
        {
            // Convert "Sentiment Text" column into numeric key types used by ML algorithm and add as a new dataset column
            // Append a classification algorithm for categorizing rows as positive or negative (binary classification)
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(PageData.PageText))
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
    }
}
