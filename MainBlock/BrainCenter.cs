using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using YOLOv4MLNet.DataStructures;
using System.Collections.Concurrent;
using System.Linq;
using ProphetLibrary;

namespace MainBlock
{
    class BrainCenter
    {
        static public ConcurrentQueue<Tuple<string, IReadOnlyList<YoloV4Result>>> yoloResults = new ConcurrentQueue<Tuple<string, IReadOnlyList<YoloV4Result>>>();
        static async Task Main(string[] args)
        {

            var cst = new CancellationTokenSource();
            var ct = cst.Token;

            Tuple<string, IReadOnlyList<YoloV4Result>> result;
            string path_for_prog = @"C:\Users\white\source\repos\BeatifulLaba\MainBlock\Assets\Images\";

            Console.WriteLine(path_for_prog);
            var cancellationTask = Task.Factory.StartNew(() =>
            {
                string cancel = Console.ReadLine();
                if (cancel == "s" || cancel == "S")
                    cst.Cancel();
            }, TaskCreationOptions.LongRunning
            );
            var task1 = Prophet.SuperImageProphet(path_for_prog, yoloResults, ct);
            var task2 = Task.Run(() =>
            {
                while (true)
                {
                    while (yoloResults.TryDequeue(out result))
                    {
                        // печать
                        var file_name = result.Item1;
                        var file_info = result.Item2;
                        var query = file_info.GroupBy(a => a.Label).Select(x => new { key = x.Key, val = x.Count() });
                        Console.WriteLine("{0} - file name\n", file_name);
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
