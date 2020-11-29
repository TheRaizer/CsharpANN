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

        static void Main(string[] args)
        {
            int[] dims = { 4, 3, 2, 1 };
            Console.Write("'predict' or 'train' or 'predict image' or 'predict iris': ");
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
            else if(choice == "predict iris")
            {
                Dictionary<string, MatrixVectors> theta = JsonConvert.DeserializeObject<Dictionary<string, MatrixVectors>>(File.ReadAllText("C:\\Users/Admin/source/repos/ANN/ANN/ThetaIris2Class.json"));

                Console.WriteLine("Test Set: ");
                IrisTrainingTest.IrisPredictions(theta, dims);
            }
        }
    }
}
