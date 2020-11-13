using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;

namespace ANN
{
    public static class FunctionTrainingTest
    {
        private static readonly List<MatrixVectors> X_training = new List<MatrixVectors>();
        private static readonly List<MatrixVectors> Y_training = new List<MatrixVectors>();
        private static readonly NeuralNetwork neuralNetwork = new NeuralNetwork(new int[] { 3, 3, 2, 1 });

        public static void NeuralNetworkSimpleFunctionTraining(int[] dims)
        {
            int numberOfTrainingExamples = 200;

            Console.WriteLine("Initializing training examples...");
            for (int i = 0; i < numberOfTrainingExamples; i++)
            {
                MatrixVectors inputVector = new MatrixVectors(dims[0], 1);
                MatrixVectors outputVector = new MatrixVectors(1, 1);
                int num = Program.rand.Next(-10, 11);
                for (int y = 0; y < dims[0]; y++)
                {
                    inputVector.MatrixVector[0, y] = num;
                }
                outputVector.MatrixVector[0, 0] = MatrixCalculations.MatrixSummation(inputVector) > 0 ? 1 : 0;
                X_training.Add(inputVector);
                Y_training.Add(outputVector);
            }

            int iterations = 800;

            Dictionary<string, MatrixVectors> theta = neuralNetwork.InitalizeParameters(dims);

            float lambda = 0.1f;
            float learningRate = 0.001f;
            float beta1 = 0.9f;
            float beta2 = 0.999f;
            float eps = 1e-8f;
            int currentStep = 0;

            Console.WriteLine("Starting network...");
            for (int i = 0; i < iterations; i++)
            {
                float cost = 0;
                for (int t = 0; t < Y_training.Count; t++)
                {
                    Tuple<List<LinearCache>, List<MatrixVectors>, MatrixVectors> cachesAndAL = neuralNetwork.ForwardPropagation(X_training[t], theta, dims);
                    cost = neuralNetwork.ComputeCost(cachesAndAL.Item3, Y_training[t], lambda, theta, dims);
                    Dictionary<string, MatrixVectors> gradients = neuralNetwork.BackwardPropagation(Y_training[t], cachesAndAL.Item3, cachesAndAL.Item1, cachesAndAL.Item2, lambda);
                    currentStep++;
                    theta = neuralNetwork.UpdateParameters(theta, gradients, dims, learningRate, beta1, beta2, currentStep, eps);
                }


                if (i % 100 == 0)
                {
                    Console.WriteLine("Cost at iteration: " + i + " = " + cost);
                    if (double.IsNaN(cost))
                    {
                        throw new ArgumentOutOfRangeException();
                    }
                }
            }

            Console.Write("Save theta? ");
            string save = Console.ReadLine();
            if (save == "yes" || save == "save")
            {
                string jsonString = JsonConvert.SerializeObject(theta, Formatting.Indented);
                File.WriteAllText("C:\\Users/Admin/source/repos/ANN/ANN/ThetaRegularized01.json", jsonString);
            }
        }

        public static void ANN_Test_Predictions(int[] dims)
        {
            Console.WriteLine("Predicting...");
            List<MatrixVectors> inputs = new List<MatrixVectors>();
            List<int> trueLabels = new List<int>();

            MatrixVectors input;
            int label;

            int numberOfPredictions = 100;

            for (int i = 0; i < numberOfPredictions; i++)
            {
                input = new MatrixVectors(dims[0], 1);
                for (int y = 0; y < dims[0]; y++)
                {
                    int num = Program.rand.Next(-10, 11);
                    input.MatrixVector[0, y] = num;
                }
                label = MatrixCalculations.MatrixSummation(input) > 0 ? 1 : 0;
                inputs.Add(input);
                trueLabels.Add(label);
            }

            Dictionary<string, MatrixVectors> theta = JsonConvert.DeserializeObject<Dictionary<string, MatrixVectors>>(File.ReadAllText("C:\\Users/Admin/source/repos/ANN/ANN/ThetaRegularized01.json"));
            Console.WriteLine("test set");
            neuralNetwork.Predict(inputs, trueLabels, theta, dims);
        }
    }
}
