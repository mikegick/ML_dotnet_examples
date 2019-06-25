using System;
using System.IO;
using System.Linq;
using IssueClassification.IssueClassification.Services;
using Microsoft.ML;

namespace IssueClassification
{
    class Program
    {
        #region fields
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        // Path to the dataset used to train the model
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
        // Path to the dataset used to evaluate the model
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
        // Path to where the trained model is saved
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");
        // MLContext object that provides processing context
        private static MLContext _mlContext;
        // IDataView object used to process the training dataset
        private static IDataView _trainingDataView;
        // PredictionEngine object used for single predictions
        private static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
        // ITransformer object that acts as the model
        private static ITransformer _trainedModel;
        #endregion

        #region dependencies
        // Service that does all processing logic for multiclass classification
        private readonly GitHubIssueClassificationService _classificationService = new GitHubIssueClassificationService();
        #endregion

        static void Main(string[] args)
        {
            // Initialize the MLContext with a random(0) seed for repeatable/deterministic results across multiple trainings
            _mlContext = new MLContext(0);

            // Load the data from text file and map columns to fields in the GitHubIssue class
            _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);

            // Extract and transform data and return processing pipeline
            var pipeline = ProcessData();

            // Build and train the model
            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);
        }

        // Extract features and transform the data into something that can be used by Microsoft.ML to train the model
        public static IEstimator<ITransformer> ProcessData()
        {
            // Transform the "Area" column into numeric key type "Label" column and add as new dataset
            // Transform text "Title" and "Description" Features into numeric vectors
            // Combine all feature columns into a new column called "Features"
            // Cache the DataView to improving performance of iterating over data multiple times
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey("Area", "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                .AppendCacheCheckpoint(_mlContext);
            // NOTE: AppendCacheCheckpoint is designed to be used for small/medium datasets to lower
            // training time and is NOT to be used when handling very large datasets

            return pipeline;
        }

        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            // Since we want to be able to predict the area of a GitHub issue as being in one of many places, we need to
            // do multiclass classification. We will use the SdcaMaximumEntropy classification training algorithm.
            // The SdcaMaximumEntropy algorithm will take in the "Label" and "Features" (Title, Description) to learn from
            // historic data and build a model from it.
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            // Visit https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.sdcamaximumentropymulticlasstrainer for
            // more info on the SdcaMaximumEntropy training algorithm

            // Train the model
            _trainedModel = trainingPipeline.Fit(trainingDataView);

            // Use the PredictionEngine convenience API to perform a single prediction
            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);

            // Use the trained model to perfom a prediction using the PredictionEngine
            // 1) Build a test issue
            GitHubIssue issue = new GitHubIssue()
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The Websockets communication used under the covers by SignalR looks like it is going slow in my" +
                "development environment."
            };

            // 2) Use the Prediction() function to make a prediction on the test issue
            var prediction = _predEngine.Predict(issue);

            // 3) Print the results of the prediction to the console
            Console.WriteLine($"Single Prediction of just-trained model on test instance: {prediction.Area}");

            // Return the trained model to use for evaluation
            return trainingPipeline;
        }
    }
}
