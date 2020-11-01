using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

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
            int[] dims = { 3, 2, 1 };
            //ANNTraining(dims);
            ANNPredictions(dims);
        }

        private static void ANNPredictions(int[] dims)
        {
            Console.WriteLine("Predicting...");
            MatrixVectors input = new MatrixVectors(dims[0], 1);
            MatrixVectors output = new MatrixVectors(1, 1);

            int numberOfPredictions = 20;

            for (int i = 0; i < numberOfPredictions; i++)
            {
                int num = rand.Next(0, 2);
                for (int y = 0; y < dims[0]; y++)
                {
                    input.MatrixVector[0, y] = num;
                }
                output.MatrixVector[0, 0] = num;
                Console.WriteLine("Expected: " + num);
                Dictionary<string, MatrixVectors> theta = JsonConvert.DeserializeObject< Dictionary<string, MatrixVectors>>(File.ReadAllText("C:\\Users/Admin/source/repos/ANN/ANN/Theta.json"));
                neuralNetwork.Predict(input, theta, dims);
            }
        }

        private static void ANNTraining(int[] dims)
        {
            int iterations = 1000;
            
            Dictionary<string, MatrixVectors> theta = neuralNetwork.InitalizeParameters(dims);

            List<MatrixVectors> X_training = new List<MatrixVectors>();
            List<MatrixVectors> Y_training = new List<MatrixVectors>();
            int numberOfTrainingExamples = 100;

            Console.WriteLine("Initializing training examples...");
            for (int i = 0; i < numberOfTrainingExamples; i++)
            {
                MatrixVectors inputVector = new MatrixVectors(dims[0], 1);
                MatrixVectors outputVector = new MatrixVectors(1, 1);
                int num = rand.Next(0, 2);
                for (int y = 0; y < dims[0]; y++)
                {
                    inputVector.MatrixVector[0, y] = num;
                }
                outputVector.MatrixVector[0, 0] = num;
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

            Console.WriteLine("Predicting0...");
            MatrixVectors input = new MatrixVectors(dims[0], 1);
            MatrixVectors output = new MatrixVectors(1, 1);
            for (int y = 0; y < dims[0]; y++)
            {
                input.MatrixVector[0, y] = 0;
            }
            output.MatrixVector[0, 0] = 0;
            Console.WriteLine("Expected: " + 0);
            neuralNetwork.Predict(input, theta, dims);

            Console.WriteLine("Predicting1...");
            for (int y = 0; y < dims[0]; y++)
            {
                input.MatrixVector[0, y] = 1;
            }
            output.MatrixVector[0, 0] = 1;
            Console.WriteLine("Expected: " + 1);
            neuralNetwork.Predict(input, theta, dims);

            string choice = Console.ReadLine();
            if(choice == "save")
            {
                string jsonString = JsonConvert.SerializeObject(theta);
                File.WriteAllText("C:\\Users/Admin/source/repos/ANN/ANN/Theta.json", jsonString);
            }
        }
    }
}
