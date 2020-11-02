using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;

namespace ANN
{
    public enum Activation
    {
       Sigmoid,
       ReLu,
       SoftMax
    }

    class Program
    {
        public readonly static Random rand = new Random();
        private static readonly NeuralNetwork neuralNetwork = new NeuralNetwork();

        static void Main(string[] args)
        {
            int[] dims = { 3, 3, 2, 1 };
            string choice = Console.ReadLine();
            if (choice == "train")
            {
                ANNTraining(dims);
            }
            else if (choice == "predict")
            {
                ANNPredictions(dims);
            }
            else
                return;
        }

        private static void ANNPredictions(int[] dims)
        {
            Console.WriteLine("Predicting...");
            MatrixVectors input = new MatrixVectors(dims[0], 1);
            MatrixVectors output = new MatrixVectors(1, 1);

            int numberOfPredictions = 20;

            for (int i = 0; i < numberOfPredictions; i++)
            {
                for (int y = 0; y < dims[0]; y++)
                {
                    int num = rand.Next(-10, 11);
                    input.MatrixVector[0, y] = num;
                }

                output.MatrixVector[0, 0] = MatrixCalculations.MatrixSummation(input) > 0 ? 1 : 0;
                Console.Write("Input: " + i);
                input.OutputMatrixValue();
                Console.WriteLine("Sum: " + MatrixCalculations.MatrixSummation(input)); 
                Console.WriteLine("Expected: " + output.MatrixVector[0, 0]);
                Dictionary<string, MatrixVectors> theta = JsonConvert.DeserializeObject< Dictionary<string, MatrixVectors>>(File.ReadAllText("C:\\Users/Admin/source/repos/ANN/ANN/Theta.json"));
                MatrixVectors outputMatrix = neuralNetwork.Predict(input, theta, dims);
                Console.WriteLine("Exact prediction: " + outputMatrix.MatrixVector[0, 0]);
            }
        }

        private static void ANNTraining(int[] dims)
        {
            int iterations = 500;
            
            Dictionary<string, MatrixVectors> theta = neuralNetwork.InitalizeParameters(dims);

            List<MatrixVectors> X_training = new List<MatrixVectors>();
            List<MatrixVectors> Y_training = new List<MatrixVectors>();
            int numberOfTrainingExamples = 150;

            Console.WriteLine("Initializing training examples...");
            for (int i = 0; i < numberOfTrainingExamples; i++)
            {
                MatrixVectors inputVector = new MatrixVectors(dims[0], 1);
                MatrixVectors outputVector = new MatrixVectors(1, 1);
                int num = rand.Next(-10, 11);
                for (int y = 0; y < dims[0]; y++)
                {
                    inputVector.MatrixVector[0, y] = num;
                }
                outputVector.MatrixVector[0, 0] = MatrixCalculations.MatrixSummation(inputVector) > 0 ? 1 : 0;
                X_training.Add(inputVector);
                Y_training.Add(outputVector);
            }

            Console.WriteLine("Starting network...");
            for (int b = 0; b < iterations; b++) 
            {
                float cost = 0;
                for (int t = 0; t < Y_training.Count; t++)
                {
                    Tuple<List<LinearCache>, List<MatrixVectors>, MatrixVectors> cachesAndAL = neuralNetwork.ForwardPropagation(X_training[t], theta, dims);
                    cost = neuralNetwork.ComputeCost(cachesAndAL.Item3, Y_training[t]);
                    Dictionary<string, MatrixVectors> gradients = neuralNetwork.BackwardPropagation(Y_training[t], cachesAndAL.Item3, cachesAndAL.Item1, cachesAndAL.Item2);
                    theta = neuralNetwork.UpdateParameters(theta, gradients, dims, 0.001f);

                }

                if (b % 100 == 0)
                    Console.WriteLine("Cost at iteration: " + b + " = " + cost);
            }

            Console.Write("Save theta? ");
            string save = Console.ReadLine();
            if(save == "yes")
            {
                string jsonString = JsonConvert.SerializeObject(theta, Formatting.Indented);
                File.WriteAllText("C:\\Users/Admin/source/repos/ANN/ANN/Theta.json", jsonString);
            }
        }
    }
}
