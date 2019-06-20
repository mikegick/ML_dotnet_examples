using System;
using System.IO;
using System.Linq;
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

        static void Main(string[] args)
        {
            
        }
    }
}
