using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace MarkDiagnosis
{
    public class PageData
    {
        [LoadColumn(0)]
        public string PageText;

        [LoadColumn(1), ColumnName("Label")]
        public bool DiagnosisExists;
    }

    public class SentimentPrediction : PageData
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
