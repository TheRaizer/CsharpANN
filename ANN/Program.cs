using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

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
        public readonly static Random rand = new Random();
        private static readonly FileLoading fileLoading = new FileLoading();
        private static readonly LoadCsvSheet csvLoading = new LoadCsvSheet();
        private static NeuralNetwork neuralNetwork;

        static void Main(string[] args)
        {
            int[] dims = { 4, 3, 2, 1 };
            neuralNetwork = new NeuralNetwork(dims);
            Console.Write("'predict' or 'train' or 'predict image': ");
            string choice = Console.ReadLine();
            if (choice == "train")
            {
                //FunctionTrainingTest.NeuralNetworkSimpleFunctionTraining(dims);
                //CatDogTrainingTest.ANNCatDogTraining(dims);
                IrisTrainingTest.IrisTraining(dims);
            }
            else if (choice == "predict")
            {
                FunctionTrainingTest.ANN_Test_Predictions(dims);
            }
            else if(choice == "predict image")
            {
                Tuple<List<MatrixVectors>, List<MatrixVectors>> data_train = fileLoading.LoadCatDogImageData(50);
                List<Tuple<List<MatrixVectors>, List<MatrixVectors>>> batches_train = neuralNetwork.GenerateBatches(50, data_train.Item1, data_train.Item2);
                Dictionary<string, MatrixVectors> theta = JsonConvert.DeserializeObject<Dictionary<string, MatrixVectors>>(File.ReadAllText("C:\\Users/Admin/source/repos/ANN/ANN/ThetaCatDog32x32.json"));
                Console.WriteLine("Training Set: ");
                CatDogTrainingTest.ANNCatDogPredictions(batches_train, theta, dims);

                Tuple<List<MatrixVectors>, List<MatrixVectors>> dataTest = fileLoading.LoadCatDogImageData(50, 3300);
                List<Tuple<List<MatrixVectors>, List<MatrixVectors>>> batchesTest = neuralNetwork.GenerateBatches(50, dataTest.Item1, dataTest.Item2);

                Console.WriteLine("Test Set: ");
                CatDogTrainingTest.ANNCatDogPredictions(batchesTest, theta, dims);
            }
            else if(choice == "predict iris")
            {
                Tuple<List<MatrixVectors>, List<MatrixVectors>> data_train = csvLoading.LoadIrisData();
                List<Tuple<List<MatrixVectors>, List<MatrixVectors>>> batches_train = neuralNetwork.GenerateBatches(32, data_train.Item1, data_train.Item2);
                Dictionary<string, MatrixVectors> theta = JsonConvert.DeserializeObject<Dictionary<string, MatrixVectors>>(File.ReadAllText("C:\\Users/Admin/source/repos/ANN/ANN/ThetaIris2Class.json"));
                Console.WriteLine("Training Set: ");
                IrisTrainingTest.IrisPredictions(batches_train, theta, dims);

                Tuple<List<MatrixVectors>, List<MatrixVectors>> data_test = csvLoading.LoadIrisData("C:\\Users/Admin/source/repos/ANN/ANN/Data/training_set/IrisData/IrisTest.csv");
                List<Tuple<List<MatrixVectors>, List<MatrixVectors>>> batches_test = neuralNetwork.GenerateBatches(32, data_test.Item1, data_test.Item2);

                Console.WriteLine("Test Set: ");
                IrisTrainingTest.IrisPredictions(batches_test, theta, dims);
            }
        }
    }
}
