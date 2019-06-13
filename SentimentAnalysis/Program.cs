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
            MLContext mLContext = new MLContext();
            TrainTestData splitDataView = LoadData(mLContext);

            // Build and train the model
            ITransformer model = BuildAndTrainModel(mLContext, splitDataView.TrainSet);
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
    }
}
