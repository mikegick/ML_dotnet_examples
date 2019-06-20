using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace IssueClassification
{
    public class GitHubIssue
    {
        [LoadColumn(0)]
        public string ID { get; set; }
        // The area of the solution where the GitHub Issue is located, this is what we'll be trying to predict
        [LoadColumn(1)]
        public string Area { get; set; }
        // GitHub issue title, the first feature used for prediction
        [LoadColumn(2)]
        public string Title { get; set; }
        // GitHub issue description, the second feature used for prediction
        [LoadColumn(3)]
        public string Description { get; set; }
    }

    public class IssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Area;
    }
}
