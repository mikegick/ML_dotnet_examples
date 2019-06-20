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
    }
}
