using Microsoft.ML;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Threading;
using System.Threading.Tasks.Dataflow;
using YOLOv4MLNet.DataStructures;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace ProphetLibrary
{
    public class Prophet
    {

        const string modelPath = @"C:\Users\white\source\repos\Lab1\Lab1\yolov4.onnx";

        static readonly string[] classesNames = new string[] { "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

        public static async Task SuperImageProphet(string path, ConcurrentQueue<Tuple<string, IReadOnlyList<YoloV4Result>>> resultsYolo, CancellationToken ct)
        {

            MLContext mlContext = new MLContext();
            var pipeline = mlContext.Transforms.ResizeImages(inputColumnName: "bitmap", outputColumnName: "input_1:0", imageWidth: 416, imageHeight: 416, resizing: ResizingKind.IsoPad)
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input_1:0", scaleImage: 1f / 255f, interleavePixelColors: true))
                .Append(mlContext.Transforms.ApplyOnnxModel(
                    shapeDictionary: new Dictionary<string, int[]>()
                    {
                        { "input_1:0", new[] { 1, 416, 416, 3 } },
                        { "Identity:0", new[] { 1, 52, 52, 3, 85 } },
                        { "Identity_1:0", new[] { 1, 26, 26, 3, 85 } },
                        { "Identity_2:0", new[] { 1, 13, 13, 3, 85 } },
                    },
                    inputColumnNames: new[]
                    {
                        "input_1:0"
                    },
                    outputColumnNames: new[]
                    {
                        "Identity:0",
                        "Identity_1:0",
                        "Identity_2:0"
                    },
                    modelFile: modelPath, recursionLimit: 100));
            var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<YoloV4BitmapData>()));
            FileInfo[] files = GetImageFiles(path);
            var images = new ConcurrentBag<string>();
            foreach (FileInfo f in files)
            {
                images.Add(f.Name);
            }

            var ab = new ActionBlock<string>(async imageName =>
            {
                using (var bitmap = new Bitmap(Image.FromFile(Path.Combine(path, imageName))))
                {
                    IReadOnlyList<YoloV4Result> results;
                    var predictionEngine = mlContext.Model.CreatePredictionEngine<YoloV4BitmapData, YoloV4Prediction>(model);
                    var predict = await AsyncGetPrediction(predictionEngine, bitmap);
                    results = predict.GetResults(classesNames, 0.3f, 0.7f);
                    Tuple<string, IReadOnlyList<YoloV4Result>> tuple = new Tuple<string, IReadOnlyList<YoloV4Result>>(imageName, results);
                    resultsYolo.Enqueue(tuple);
                }
            },
            new ExecutionDataflowBlockOptions
            {
                MaxDegreeOfParallelism = 4,
                CancellationToken = ct
            });

            Parallel.ForEach(images, imageName => ab.Post(imageName));
            ab.Complete();

            try
            {
                await ab.Completion;
            }
            catch (TaskCanceledException)
            {
                Console.WriteLine("Abort successful");
            }
            catch
            {
                Console.WriteLine("Unexpected behaviour");
            }
        }
        static async Task<YoloV4Prediction> AsyncGetPrediction(PredictionEngine<YoloV4BitmapData, YoloV4Prediction> predictionEngine, Bitmap bitmap)
        {
            return await Task<YoloV4Prediction>.Factory.StartNew(() =>
            {
                return predictionEngine.Predict(new YoloV4BitmapData() { Image = bitmap }); ;
            });
        }
        
        public static FileInfo[] GetImageFiles(string path)
        {
            DirectoryInfo dir = new DirectoryInfo(path);
            string[] extensions = new[] { ".jpg", ".jpeg", ".tiff", ".bmp", ".png" };
            FileInfo[] files = dir.GetFiles().Where(f => extensions.Contains(f.Extension.ToLower())).ToArray();
            return files;
        }
    }
}
