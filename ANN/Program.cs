using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;

namespace ANN
{
    //theta.json and thetaRegularized0001.json dims = {3, 3, 2, 1}
    //thetaCatDog32x32.json and thetaCatDog32x32Regularized.json dims = {3072, 5, 3, 1}
    //thetaCatDog64x64.json dims = {12288, 20, 7, 5, 1}

    public enum Activation
    {
       Sigmoid,
       ReLu,
       SoftMax
    }

    class Program
    {
        private static readonly List<MatrixVectors> X_training = new List<MatrixVectors>();
        private static readonly List<MatrixVectors> Y_training = new List<MatrixVectors>();
        public readonly static Random rand = new Random();
        private static readonly NeuralNetwork neuralNetwork = new NeuralNetwork();

        private static readonly FileLoading fileLoading = new FileLoading();

        static void Main(string[] args)
        {
            int[] dims = { 3072, 5, 3, 1 };
            Console.Write("'predict' or 'train' or 'predict image': ");
            string choice = Console.ReadLine();
            if (choice == "train")
            {
                //NeuralNetworkTestTraining(dims);
                ANNTraining(dims);
            }
            else if (choice == "predict")
            {
                ANN_Test_Predictions(dims);
            }
            else if(choice == "predict image")
            {
                Tuple<List<MatrixVectors>, List<MatrixVectors>> data = fileLoading.LoadCatDogImageData(50);
                List<Tuple<List<MatrixVectors>, List<MatrixVectors>>> batches = neuralNetwork.GenerateBatches(50, data.Item1, data.Item2);
                Dictionary<string, MatrixVectors> theta = JsonConvert.DeserializeObject<Dictionary<string, MatrixVectors>>(File.ReadAllText("C:\\Users/Admin/source/repos/ANN/ANN/ThetaCatDog32x32.json"));
                Console.WriteLine(theta["W" + 1].Shape());
                Console.WriteLine(batches[0].Item1[0].Shape());
                Console.WriteLine("Training Set: ");
                ANNCatDogPredictions(batches, theta, dims);

                Console.WriteLine("Test Set: ");
                Tuple<List<MatrixVectors>, List<MatrixVectors>> dataTest = fileLoading.LoadCatDogImageData(50, 3300);
                List<Tuple<List<MatrixVectors>, List<MatrixVectors>>> batchesTest = neuralNetwork.GenerateBatches(50, dataTest.Item1, dataTest.Item2);
                ANNCatDogPredictions(batchesTest, theta, dims);
            }
        }

        private static void ANN_Test_Predictions(int[] dims)
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
                    int num = rand.Next(-10, 11);
                    input.MatrixVector[0, y] = num;
                }
                label = MatrixCalculations.MatrixSummation(input) > 0 ? 1 : 0;
                inputs.Add(input);
                trueLabels.Add(label);
            }

            Dictionary<string, MatrixVectors> theta = JsonConvert.DeserializeObject<Dictionary<string, MatrixVectors>>(File.ReadAllText("C:\\Users/Admin/source/repos/ANN/ANN/Theta.json"));
            Console.WriteLine("test set");
            neuralNetwork.Predict(inputs, trueLabels, theta, dims);
        }

        private static void ANNCatDogPredictions(List<Tuple<List<MatrixVectors>, List<MatrixVectors>>> batches, Dictionary<string, MatrixVectors> theta, int[] dims)
        {
            List<int> labels = new List<int>();
            for (int i = 0; i < batches[0].Item2.Count; i++)
            {
                labels.Add((int)batches[0].Item2[i].MatrixVector[0, 0]);
            }
            neuralNetwork.Predict(batches[0].Item1, labels, theta, dims);   
        }

        private static void ANNTraining(int[] dims)
        {
            int iterations = 2500;
            
            Dictionary<string, MatrixVectors> theta = neuralNetwork.InitalizeParameters(dims);
            Tuple<List<MatrixVectors>, List<MatrixVectors>> data = fileLoading.LoadCatDogImageData(1400);
            List<Tuple<List<MatrixVectors>, List<MatrixVectors>>> batches = neuralNetwork.GenerateBatches(32, data.Item1, data.Item2);

            float lambda = 0f;
            float learningRate = 0.0075f;

            Console.WriteLine("Starting network...");
            for (int i = 0; i < iterations; i++)
            {
                float cost = 0;
                Console.WriteLine(i);
                for (int b = 0; b < batches.Count; b++)
                {
                    List<MatrixVectors> XBatch = batches[b].Item1;
                    List<MatrixVectors> YBatch = batches[b].Item2;
                    for (int t = 0; t < XBatch.Count; t++)
                    {
                        Tuple<List<LinearCache>, List<MatrixVectors>, MatrixVectors> cachesAndAL = neuralNetwork.ForwardPropagation(XBatch[t], theta, dims);
                        cost = neuralNetwork.ComputeCost(cachesAndAL.Item3, YBatch[t], lambda, theta, dims);
                        Dictionary<string, MatrixVectors> gradients = neuralNetwork.BackwardPropagation(YBatch[t], cachesAndAL.Item3, cachesAndAL.Item1, cachesAndAL.Item2, lambda);
                        theta = neuralNetwork.UpdateParameters(theta, gradients, dims, learningRate);
                    }
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
            ANNCatDogPredictions(batches, theta, dims);

            Console.Write("Save theta? ");
            string save = Console.ReadLine();
            if(save == "yes" || save == "save")
            {
                string jsonString = JsonConvert.SerializeObject(theta, Formatting.Indented);
                File.WriteAllText("C:\\Users/Admin/source/repos/ANN/ANN/ThetaCatDog32x32.json", jsonString);
            }
        }

        private static void NeuralNetworkTestTraining(int[] dims)
        {
            int numberOfTrainingExamples = 200;

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

            int iterations = 800;

            Dictionary<string, MatrixVectors> theta = neuralNetwork.InitalizeParameters(dims);

            float lambda = 0f;
            float learningRate = 0.001f;

            Console.WriteLine("Starting network...");
            for (int i = 0; i < iterations; i++)
            {
                float cost = 0;
                for (int t = 0; t < Y_training.Count; t++)
                {
                    Tuple<List<LinearCache>, List<MatrixVectors>, MatrixVectors> cachesAndAL = neuralNetwork.ForwardPropagation(X_training[t], theta, dims);
                    cost = neuralNetwork.ComputeCost(cachesAndAL.Item3, Y_training[t], lambda, theta, dims);
                    Dictionary<string, MatrixVectors> gradients = neuralNetwork.BackwardPropagation(Y_training[t], cachesAndAL.Item3, cachesAndAL.Item1, cachesAndAL.Item2, lambda);
                    theta = neuralNetwork.UpdateParameters(theta, gradients, dims, learningRate);
                }
                

                if (i % 100 == 0)
                    Console.WriteLine("Cost at iteration: " + i + " = " + cost);
            }

            Console.Write("Save theta? ");
            string save = Console.ReadLine();
            if (save == "yes")
            {
                string jsonString = JsonConvert.SerializeObject(theta, Formatting.Indented);
                File.WriteAllText("C:\\Users/Admin/source/repos/ANN/ANN/ThetaRegularized0001.json", jsonString);
            }
        }
    }
}
