using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;

namespace ANN
{
    public static class CatDogTrainingTest
    {
        private static readonly NeuralNetwork neuralNetwork = new NeuralNetwork(new int[] { 3072, 5, 3, 1 });
        private static readonly FileLoading fileLoading = new FileLoading();

        public static void ANNCatDogPredictions(List<Tuple<List<MatrixVectors>, List<MatrixVectors>>> batches, Dictionary<string, MatrixVectors> theta, int[] dims)
        {
            List<int> labels = new List<int>();
            for (int i = 0; i < batches[0].Item2.Count; i++)
            {
                labels.Add((int)batches[0].Item2[i].MatrixVector[0, 0]);
            }
            neuralNetwork.Predict(batches[0].Item1, labels, theta, dims);
        }

        public static void ANNCatDogTraining(int[] dims)
        {
            int iterations = 2500;

            Dictionary<string, MatrixVectors> theta = neuralNetwork.InitalizeParameters(dims);
            Tuple<List<MatrixVectors>, List<MatrixVectors>> data = fileLoading.LoadCatDogImageData(1400);
            List<Tuple<List<MatrixVectors>, List<MatrixVectors>>> batches = neuralNetwork.GenerateBatches(32, data.Item1, data.Item2);

            float lambda = 0f;
            float learningRate = 0.0075f;
            float beta1 = 0.9f;
            float beta2 = 0.999f;
            float eps = 1e-8f;
            int currentStep = 0;

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
                        currentStep++;
                        theta = neuralNetwork.UpdateParameters(theta, gradients, dims, learningRate, beta1, beta2, currentStep, eps);
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
            if (save == "yes" || save == "save")
            {
                string jsonString = JsonConvert.SerializeObject(theta, Formatting.Indented);
                File.WriteAllText("C:\\Users/Admin/source/repos/ANN/ANN/ThetaCatDog32x32.json", jsonString);
            }
        }
    }
}
