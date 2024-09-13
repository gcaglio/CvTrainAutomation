using System;
using System.Collections;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using AForge.Video.DirectShow;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

class Program
{

    static int frame_index=0;
    static Hashtable hook_sent;

    

    static async Task Main(string[] args)
    {
        // Carica configurazione
        var config = LoadConfiguration("config.json");

        // Carica i pattern da un file di configurazione
        var patterns = LoadPatternConfiguration("config_pattern.json");
        //PatternMat[] templates = LoadTemplates(patterns);

        // init hashtable per gestire gli hook
        hook_sent = new Hashtable();

        // Inizializza le webcam
        var videoDevices = new FilterInfoCollection(FilterCategory.VideoInputDevice);
        for (int v=0; v < videoDevices.Count;v++)
        {
            var device = videoDevices[v];
            Console.WriteLine("INFO : detected webcam '" + videoDevices[v].Name + "' with index '"+v+"'");
        }



        foreach (var webcamConfig in config.Webcams)
        {
            if (webcamConfig.StreamIndex < videoDevices.Count)
            {

                hook_sent[webcamConfig.Id] = null;      //inizializzo con NULL

                var videoSource = new VideoCaptureDevice(videoDevices[webcamConfig.StreamIndex].MonikerString);
                videoSource.NewFrame += (sender, eventArgs) => ProcessFrame(eventArgs.Frame, webcamConfig, patterns);

                // Set resolution and fps
                var videoCapabilities = videoSource.VideoCapabilities;
                var selectedCapability = Array.Find(videoCapabilities,
                    cap => cap.FrameSize.Width == webcamConfig.Width &&
                           cap.FrameSize.Height == webcamConfig.Height);

                if (selectedCapability != null)
                {
                    videoSource.VideoResolution = selectedCapability;
                    Console.WriteLine($"INFO : Webcam con StreamIndex {webcamConfig.StreamIndex} - selected Resolution: {selectedCapability.FrameSize.Width}x{selectedCapability.FrameSize.Height}");

                    selectedCapability = Array.Find(videoCapabilities,
                         cap => cap.FrameSize.Width == webcamConfig.Width &&
                         cap.FrameSize.Height == webcamConfig.Height &&
                         cap.AverageFrameRate == webcamConfig.Fps);

                    if (selectedCapability != null)
                    {
                        videoSource.VideoResolution = selectedCapability;
                        Console.WriteLine($"INFO : Webcam con StreamIndex {webcamConfig.StreamIndex} - selected Framerate: {selectedCapability.AverageFrameRate} fps");
                    }
                    else
                    {
                        Console.WriteLine($"ERROR : Webcam con StreamIndex {webcamConfig.StreamIndex} - il framerate {webcamConfig.Fps} desiderati non è disponibile");
                        foreach (var capability in videoCapabilities)
                        {
                            Console.WriteLine($"Framerate: {capability.AverageFrameRate} fps");
                        }
                        Console.WriteLine();
                    }
                }
                else
                {
                    Console.WriteLine($"ERROR : Webcam con StreamIndex {webcamConfig.StreamIndex} - la risoluzione {webcamConfig.Width}x{webcamConfig.Height} non è disponibilw");
                    foreach (var capability in videoCapabilities)
                    {
                        Console.WriteLine($"Resolution: {capability.FrameSize.Width}x{capability.FrameSize.Height}");
                    }
                    Console.WriteLine();
                }


   


                // Set focus
                try
                {
                    int focusValue = webcamConfig.Focus;
                    videoSource.SetCameraProperty(AForge.Video.DirectShow.CameraControlProperty.Focus, focusValue, AForge.Video.DirectShow.CameraControlFlags.Manual);
                    Console.WriteLine($"INFO : Webcam con StreamIndex {webcamConfig.StreamIndex} - fuoco impostato a {focusValue}.");
                }
                catch (Exception e)
                {
                    Console.WriteLine($"ERROR : Webcam con StreamIndex {webcamConfig.StreamIndex} - La webcam non supporta la regolazione del fuoco.");
                }

                videoSource.Start();
            }
            else
            {
                Console.WriteLine($"Webcam con StreamIndex {webcamConfig.StreamIndex} non trovata.");
            }
        }

        Console.WriteLine("Premi un tasto per uscire...");
        Console.ReadKey();
    }

    static PatternMat[] LoadTemplates(PatternsConfig patterns)
    {
        int valid_pictures = 0;
        foreach (var pattern in patterns.Patterns)
        {

            FileInfo f = new FileInfo(pattern.Path);
            if (!f.Exists)
            {
                Console.WriteLine("ERROR : pattern '" + pattern.Path + "' does not exists!");
            }
            valid_pictures++;

        }

        PatternMat[] templates = new  PatternMat[valid_pictures];

        int c = 0;  
        foreach (var pattern in patterns.Patterns)
        {

            FileInfo f = new FileInfo(pattern.Path);
            if (!f.Exists)
            {
                Console.WriteLine("ERROR : pattern '" + pattern.Path + "' does not exists!");
                continue;
            }
            
            using var template = new Image<Gray, byte>(pattern.Path).Mat;
            PatternMat t = new PatternMat();
            t.Mat = template;
            t.Description = pattern.Description;
            t.Id = pattern.Id;

            templates[c] = t;
            c++;
        }

        return templates;

    }

    static void ProcessFrame(Bitmap frame, WebcamConfig config, PatternsConfig patterns)
    {
        
        if (frame_index % 3 != 0)
        {
            frame_index++;
            return;
        }
        else
        {
            frame_index = 0;
        }
        



        using var mat = BitmapToMat(frame);
        var grayMat = new Mat();
        //CvInvoke.CvtColor(mat, grayMat, ColorConversion.Bgr2Gray);
        CvInvoke.CvtColor(mat, grayMat, ColorConversion.Bgr2Rgb);
        Confidence[] confidences = new Confidence[patterns.Patterns.Count()];
        int c = 0;

        foreach (var pattern in patterns.Patterns)
        {

            FileInfo f = new FileInfo(pattern.Path);
            if (!f.Exists)
            {
                Console.WriteLine("ERROR : pattern '"+pattern.Path+"' does not exists!");
            }

            //using var template = new Image<Gray, byte>(pattern.Path).Mat;
            using var template = new Image<Emgu.CV.Structure.Rgb, byte>(pattern.Path).Mat;
            
            using var result = new Mat();

            // Confronto tra il frame attuale e il pattern
            CvInvoke.MatchTemplate(grayMat, template, result, TemplateMatchingType.CcoeffNormed);
            

            result.MinMax(out double[] minValues, out double[] maxValues, out Point[] minLocations, out Point[] maxLocations);

            //save confidence
            confidences[c] = new Confidence();
            confidences[c].Patternconfig = pattern;
            confidences[c].ConfPercentage = maxValues[0];
            confidences[c].hook_sent_timestamp = DateTime.MaxValue ;        //set to max value to avoid first false positive


            // Se il valore di confidenza è alto, segnala la corrispondenza
            //Console.WriteLine($"DEBUG [Webcam ID: {config.Id}] Pattern '{pattern.Description}' trovato con confidenza {maxValues[0]:F2}");
            /*
            if (maxValues[0] >= 0.4)
            {
            //    Console.WriteLine($"PASSED [Webcam ID: {config.Id}] Pattern '{pattern.Description}' trovato con confidenza {maxValues[0]:F2}");
                var rect = new Rectangle(maxLocations[0], template.Size);
                CvInvoke.Rectangle(mat, rect, new MCvScalar(0, 255, 0), 3);
            }
            */
            //CvInvoke.Imshow($"Webcam {config.Id}", mat);
            
            c++;            
        }

        Confidence current_max = confidences[0];
        current_max.isMax = true;
        for (int m = 1; m < confidences.Length; m++)
        {
            if (current_max.ConfPercentage < confidences[m].ConfPercentage)
            {
                current_max.isMax = false;
                
                current_max = confidences[m];
                current_max.isMax = true;
            }
        }

      

        Console.WriteLine("DEBUG : results");
        double threshold = 0.7;
        foreach (Confidence confidence in confidences)
        {
            string selector = " ";
            if (confidence.isMax)
                selector = ">";

            string choosen = " ";
            if (confidence.ConfPercentage>threshold)
                choosen = "*";


            Console.WriteLine("       "+choosen+selector + confidence.Patternconfig.Description + " with confidence " + confidence.ConfPercentage);


            // chiamo l'hook e aggiorno la hashtable
            if (confidence.isMax && confidence.ConfPercentage > threshold)
            {
                if (  ((Confidence)hook_sent[config.Id]).hook_sent_timestamp   )
                hook_sent[config.Id] = confidence;
                confidence.hook_sent_timestamp = DateTime.Now;
            }


        }
        Console.SetCursorPosition(0, Console.CursorTop - confidences.Count() -1);
        /*
        
        foreach (var pattern in templates)
        {


            using var template = pattern.Mat;
            using var result = new Mat();
            
            // Confronto tra il frame attuale e il pattern
            CvInvoke.MatchTemplate(grayMat, template, result, TemplateMatchingType.CcoeffNormed);
            
            LoadTemplates(patterns);

            result.MinMax(out double[] minValues, out double[] maxValues, out Point[] minLocations, out Point[] maxLocations);


            // Se il valore di confidenza è alto, segnala la corrispondenza
            //Console.WriteLine($"DEBUG [Webcam ID: {config.Id}] Pattern '{pattern.Description}' trovato con confidenza {maxValues[0]:F2}");
            if (maxValues[0] >= 0.4)
            {
                Console.WriteLine($"PASSED [Webcam ID: {config.Id}] Pattern '{pattern.Description}' trovato con confidenza {maxValues[0]:F2}");
                var rect = new Rectangle(maxLocations[0], template.Size);
                CvInvoke.Rectangle(mat, rect, new MCvScalar(0, 255, 0), 3);
            }
        } */

        //CvInvoke.Imshow($"Webcam {config.Id}", mat);

        //if (CvInvoke.WaitKey(1) > 0)
        //{
        //    Environment.Exit(0);
        //}
    }

    static Mat BitmapToMat(Bitmap bitmap)
    {
        var mat = new Mat(bitmap.Height, bitmap.Width, DepthType.Cv8U, 3);
        var bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
        try
        {
            var ptr = bitmapData.Scan0;
            var bytes = new byte[bitmapData.Stride * bitmap.Height];
            System.Runtime.InteropServices.Marshal.Copy(ptr, bytes, 0, bytes.Length);
            mat.SetTo(bytes);
        }
        finally
        {
            bitmap.UnlockBits(bitmapData);
        }
        return mat;
    }

    static PatternsConfig LoadPatternConfiguration(string path)
    {
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<PatternsConfig>(json);
    }

    static async Task SendDataToServer(string webcamId, string qrCodeData, string url)
    {
        using var client = new HttpClient();
        try
        {
            Console.WriteLine($"INFO : sending string '{qrCodeData}' from webcam '{webcamId}'.\r\n      {url}");
            var response = await client.GetAsync(url);
            response.EnsureSuccessStatusCode();
            Console.WriteLine("INFO : Dati inviati consuccesso.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Errore nell'invio dei dati: {ex.Message}");
        }
    }

    static string FormatUrl(string urlTemplate, string webcamId, string qrCodeData)
    {
        return urlTemplate
            .Replace("{webcamId}", Uri.EscapeDataString(webcamId))
            .Replace("{qrCodeData}", Uri.EscapeDataString(qrCodeData));
    }

    static AppConfig LoadConfiguration(string path)
    {
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<AppConfig>(json);
    }
}

public class AppConfig
{
    public WebcamConfig[] Webcams { get; set; }
}

public class PatternsConfig
{
    public PatternConfig[] Patterns { get; set; }
}

public class WebcamConfig
{
    public string Id { get; set; }
    public int StreamIndex { get; set; }
    public string Url { get; set; }
    public int Width { get; set; }   // Larghezza desiderata
    public int Height { get; set; }  // Altezza desiderata
    public int Fps { get; set; }     // Framerate desiderato
    public int Focus { get; set; }   // Focus 1-100 (1 infinito, 100 vicino)
}

public class PatternConfig
{
    public int Id { get; set; }
    public string Description { get; set; }
    public string Path { get; set; }
}

public class PatternMat
{
    public Mat Mat { get; set; }
    public int Id { get; set; }
    public string Description { get; set; }
}

public class Confidence
{
    public PatternConfig Patternconfig { get; set; }
    public double ConfPercentage { get; set; }
    public bool isMax { get; set; }

    public DateTime hook_sent_timestamp { get; set; }
}

