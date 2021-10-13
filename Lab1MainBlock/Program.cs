using System;
using System.Collections.Generic;
using LabMainBlock;
using YOLOv4MLNet;
using System.Threading;
using System.Threading.Tasks;
using YOLOv4MLNet.DataStructures;
using System.Collections.Concurrent;
using System.Linq;

namespace Lab1MainBlock
{
    class Program
    {
        static public ConcurrentQueue<Tuple<string, IReadOnlyList<YoloV4Result>>> yoloResults = new ConcurrentQueue<Tuple<string, IReadOnlyList<YoloV4Result>>>();
        static public List<string> labelsList = new List<string>();
        static async Task Main(string[] args)
        {

            var cst = new CancellationTokenSource();
            var ct = cst.Token;

            Tuple<string, IReadOnlyList<YoloV4Result>> result;
            string path_for_prog = @"Assets\Images";

            var cancellationTask = Task.Run(() =>
            {
                string cancel = Console.ReadLine();
                if (cancel == "s" || cancel == "S")
                    cst.Cancel();
            }
            );
            var task1 = ImageProcessing.SuperImageProphet(path_for_prog, yoloResults, ct);
            var task2 = Task.Run(() =>
            {
                while (true)
                {
                    while (yoloResults.TryDequeue(out result))
                    {
                        // печать
                        var file_name = result.Item1;
                        var file_info = result.Item2;

                        foreach (var object_new in file_info)
                        {
                            labelsList.Add(object_new.Label);
                        }
                        var query = labelsList.GroupBy(a => a).Select(x => new { key = x.Key, val = x.Count()});
                        Console.WriteLine("{0} - file name",file_name);
                        foreach (var result in query)
                        {
                            Console.WriteLine("{0} - object_name, {1} - count", result.key, result.val);
                        }
                    }
                }

            });
            await Task.WhenAll(task1);
        }
    }
}
