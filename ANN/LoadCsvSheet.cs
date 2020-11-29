using System;
using System.Collections.Generic;
using System.IO;

namespace ANN
{
    public class LoadCsvSheet
    {
        public Tuple<List<MatrixVectors>, List<MatrixVectors>> LoadIrisData(string path = "C:\\Users/Admin/source/repos/ANN/ANN/Data/IrisData/IrisTraining.csv")
        {
            List<MatrixVectors> dataVector = new List<MatrixVectors>();
            List<MatrixVectors> labelVector = new List<MatrixVectors>();

            using (StreamReader labeledReader = new StreamReader(path))
            {
                int k = 0;
                while (!labeledReader.EndOfStream)
                {
                    k++;
                    string line = labeledReader.ReadLine();
                    if (k != 1)
                    {
                        string[] values = line.Split(',');
                        List<float> currentData = new List<float>();
                        MatrixVectors currentLabel = new MatrixVectors(1, 1);
                        for (int i = 1; i < values.Length; i++)
                        {
                            if(i == 5)
                            {
                                if (values[i] == "Iris-setosa")
                                {
                                    currentLabel.MatrixVector[0, 0] = 0;
                                }
                                else if (values[i] == "Iris-versicolor")
                                {
                                    currentLabel.MatrixVector[0, 0] = 1;
                                }
                            }
                            else
                            {
                                currentData.Add(float.Parse(values[i]));
                            }
                        }
                        dataVector.Add(currentData.ListToVector(1));
                        labelVector.Add(currentLabel);
                    }
                }
            }
            return new Tuple<List<MatrixVectors>, List<MatrixVectors>>(dataVector, labelVector);
        }
    }
}
