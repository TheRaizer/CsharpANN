using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;

namespace ANN
{
    public class FileLoading
    {
        public Tuple<List<MatrixVectors>, List<MatrixVectors>> LoadCatDogImageData(int numberOfImages, int startingIndex = 0)
        {
            ///<summary>
            /// This only works if the images in the file are all
            /// labeled the same thing.
            ///</summary>
            
            Console.WriteLine("Started Loading");
            string[] catFiles = Directory.GetFiles("C:\\Users/Admin/source/repos/ANN/ANN/Data/training_set/CatsImages32x32/CatPics/");
            string[] dogFiles = Directory.GetFiles("C:\\Users/Admin/source/repos/ANN/ANN/Data/training_set/DogsImages32x32/DogPics/");
            List<MatrixVectors> data = new List<MatrixVectors>();
            List<MatrixVectors> labels = new List<MatrixVectors>();

            for(int i = startingIndex; i < numberOfImages + startingIndex; i++)
            {
                Bitmap catImage = new Bitmap(catFiles[i]);
                Bitmap dogImage = new Bitmap(dogFiles[i]);
                List<float> catPixelValues = new List<float>();
                List<float> dogPixelValues = new List<float>();

                for (int x = 0; x < catImage.Width; x++)
                {
                    for(int y = 0; y < catImage.Height; y++)
                    {
                        catPixelValues.Add((float)catImage.GetPixel(x, y).R / 255);
                        catPixelValues.Add((float)catImage.GetPixel(x, y).G / 255);
                        catPixelValues.Add((float)catImage.GetPixel(x, y).B / 255);

                        dogPixelValues.Add((float)dogImage.GetPixel(x, y).R / 255);
                        dogPixelValues.Add((float)dogImage.GetPixel(x, y).G / 255);
                        dogPixelValues.Add((float)dogImage.GetPixel(x, y).B / 255);
                    }
                }
                MatrixVectors catVector = MatrixCalculations.ListToVector(catPixelValues, 1);
                MatrixVectors dogVector = MatrixCalculations.ListToVector(dogPixelValues, 1);

                MatrixVectors catLabel = new MatrixVectors(1, 1);
                MatrixVectors dogLabel = new MatrixVectors(1, 1);

                catLabel.MatrixVector[0, 0] = 0;
                dogLabel.MatrixVector[0, 0] = 1;

                labels.Add(catLabel);
                labels.Add(dogLabel);
                data.Add(dogVector);
                data.Add(catVector);
            }
            Console.WriteLine("Finished Loading");
            return new Tuple<List<MatrixVectors>, List<MatrixVectors>>(data, labels);
        }
    }
}
