using AForge.Video.DirectShow;
using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using QrTrainAutomation;
using System.Collections;
using System.Drawing;
using System.Drawing.Imaging;
using System.Text;
using System.Text.Json;

class Program
{

    static int frame_index = 0;
    static Hashtable hook_sent;
    static bool gpuAvailable = false;


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
        for (int v = 0; v < videoDevices.Count; v++)
        {
            var device = videoDevices[v];
            Console.WriteLine("INFO : detected webcam '" + videoDevices[v].Name + "' with index '" + v + "'");
        }



        foreach (var webcamConfig in config.Webcams)
        {
            if (webcamConfig.StreamIndex < videoDevices.Count)
            {

               

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

                //Thread.Sleep(2000);
                videoSource.Start();
            }
            else
            {
                Console.WriteLine($"Webcam con StreamIndex {webcamConfig.StreamIndex} non trovata.");
            }
        }


        // Verifica se la GPU è disponibile
        gpuAvailable = CudaInvoke.HasCuda;
        Console.WriteLine("INFO : system has gpu/cuda = " + gpuAvailable);
        
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

        PatternMat[] templates = new PatternMat[valid_pictures];

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
        int cooldown_msec = 13000;

        if (frame_index % 2 != 0)
        {
            frame_index++;
            return;
        }
        else
        {
            frame_index = 0;
            // Controllo se la chiave esiste nella hashtable
            if (hook_sent.ContainsKey(config.Id) && hook_sent[config.Id]!=null)
            {
                Confidence previousConfidence = (Confidence)hook_sent[config.Id];

                // Controllo se il campo hook_sent_timestamp è valorizzato e se è < NOW
                if (previousConfidence.hook_sent_timestamp != null && ((long)(DateTime.Now - previousConfidence.hook_sent_timestamp).TotalMilliseconds > cooldown_msec) )
                {
                    hook_sent.Remove(config.Id);
                }
                else
                {
                    Console.WriteLine($"Skipping actions for webcam : " + config.Id + " due to cooldown (" + (cooldown_msec - (long)(DateTime.Now - previousConfidence.hook_sent_timestamp).TotalMilliseconds) + " msec).");
                    return;  // Se il timestamp è recente, salta l'esecuzione degli hook
                }
            }

        }





        using var mat = BitmapToMat(frame); // Converte il bitmap in Mat per OpenCV

        CvInvoke.Resize(mat, mat, new Size(mat.Width / 2, mat.Height / 2));
        



        Confidence[] confidences = new Confidence[patterns.Patterns.Count()];
        



        // Parallelizza l'elaborazione di ciascun pattern
        Parallel.ForEach(patterns.Patterns, (pattern, state, index) =>
        {
            FileInfo f = new FileInfo(pattern.Path);
            if (!f.Exists)
            {
                Console.WriteLine($"ERROR : pattern '{pattern.Path}' does not exist!");
                return;
            }

            // Carica il template del pattern corrente
            var template = new Image<Bgr, byte>(pattern.Path).Mat;
            CvInvoke.Resize(template, template, new Size(template.Width / 2, template.Height / 2));
            if (template.IsEmpty)
            {
                Console.WriteLine($"ERROR : pattern '{pattern.Path}' is empty!");
                return;
            }

            // Se la GPU è disponibile, puoi usare operazioni GPU prima di MatchTemplate
            if (gpuAvailable)
            {
                // Esempio: puoi caricare l'immagine e il template su GPU per eventuali pre-processing
                using var gpuFrame = new CudaImage<Bgr, byte>(mat);
                using var gpuTemplate = new CudaImage<Bgr, byte>(template);

                // Esegui operazioni GPU-accelerate come ridimensionamento o conversioni di colore
                // gpuFrame.Convert<Gray, byte>(); // Esempio di operazione sulla GPU

                // Al termine delle operazioni GPU, scarica le immagini processate dalla GPU alla CPU
                gpuFrame.Download(mat);
                gpuTemplate.Download(template);
            }

            // Esegui il template matching sulla CPU (in quanto non disponibile sulla GPU)
            using Mat result = new Mat();
            CvInvoke.MatchTemplate(mat, template, result, Emgu.CV.CvEnum.TemplateMatchingType.CcoeffNormed);

            // Estrai i risultati del matching
            result.MinMax(out double[] minValues, out double[] maxValues, out Point[] minLocations, out Point[] maxLocations);

            // Salva la confidenza corrente
            confidences[(int)index] = new Confidence
            {
                Patternconfig = pattern,
                ConfPercentage = maxValues[0],
                hook_sent_timestamp = DateTime.Now
            };
        });

        // Trova il pattern con la confidenza massima
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

        // Stampa i risultati e controlla i trigger
        Console.WriteLine("DEBUG : results");
        double threshold = 0.6;
        foreach (Confidence confidence in confidences)
        {
            string selector = confidence.isMax ? ">" : " ";
            string choosen = confidence.ConfPercentage > threshold ? "*" : " ";
            

            Console.WriteLine($"{choosen}{selector} {confidence.Patternconfig.Description} with confidence {confidence.ConfPercentage}");



            if (confidence.isMax && confidence.ConfPercentage > threshold)
            {
                Console.WriteLine($"Triggering actions for pattern '{confidence.Patternconfig.Description}'");

                // Aggiorna la hashtable con l'oggetto Confidence aggiornato
                hook_sent[config.Id] = confidence;


                // Esegui le azioni nei hooks in un thread parallelo
                //Thread hookThread = new Thread(() => ExecuteHooks(config.Hooks, confidence.Patternconfig.Description));
                //hookThread.Start();

                // Esegui le azioni nei hooks del pattern, non più dal config della webcam
                if (confidence.Patternconfig.Hooks != null)
                {
                    Thread hookThread = new Thread(() => ExecuteHooks(confidence.Patternconfig.Hooks, confidence.Patternconfig.Description));
                    hookThread.Start();
                }
            }
        }

        // Riposiziona il cursore della console per un output leggibile
        Console.SetCursorPosition(0, Console.CursorTop - confidences.Length - 1);
    }

    static void ProcessFrame2(Bitmap frame, WebcamConfig config, PatternsConfig patterns)
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
        //var grayMat = new Mat();
        //CvInvoke.CvtColor(mat, grayMat, ColorConversion.Bgr2Gray);
        //CvInvoke.CvtColor(mat, grayMat, ColorConversion.Bgr2Rgb);
        Confidence[] confidences = new Confidence[patterns.Patterns.Count()];
        int c = 0;


        Parallel.For(0, patterns.Patterns.Length, (i) =>
        {
            var pattern = patterns.Patterns[i];
            // Carica il template e confronta
            var template = new Image<Bgr, byte>(pattern.Path).Mat;
            if (!template.IsEmpty)
            {
                using var result = new Mat();
                CvInvoke.MatchTemplate(mat, template, result, TemplateMatchingType.CcoeffNormed);
                result.MinMax(out double[] minValues, out double[] maxValues, out Point[] minLocations, out Point[] maxLocations);

                lock (confidences) // sincronizza l'accesso a "confidences" per la scrittura in parallelo
                {
                    confidences[i] = new Confidence
                    {
                        Patternconfig = pattern,
                        ConfPercentage = maxValues[0],
                        hook_sent_timestamp = DateTime.MaxValue
                    };
                }
            }
        });


        /* NON PARALLELO
        foreach (var pattern in patterns.Patterns)
        {

            FileInfo f = new FileInfo(pattern.Path);
            if (!f.Exists )
            {
                Console.WriteLine("ERROR : pattern '" + pattern.Path + "' does not exists!");
            }

            //using var template = new Image<Gray, byte>(pattern.Path).Mat;
            var template = new Image<Emgu.CV.Structure.Bgr, byte>(pattern.Path).Mat;
            if (template.IsEmpty)
            {
                Console.WriteLine("ERROR : pattern '" + pattern.Path + "' is empty!");
                continue;
            }
            using var result = new Mat();

            // Confronto tra il frame attuale e il pattern
            //CvInvoke.MatchTemplate(grayMat, template, result, TemplateMatchingType.CcoeffNormed);
            CvInvoke.MatchTemplate( mat, template, result, TemplateMatchingType.CcoeffNormed);

            result.MinMax(out double[] minValues, out double[] maxValues, out Point[] minLocations, out Point[] maxLocations);

            //save confidence       
            confidences[c] = new Confidence();
            confidences[c].Patternconfig = pattern;
            confidences[c].ConfPercentage = maxValues[0];
            confidences[c].hook_sent_timestamp = DateTime.MaxValue;        //set to max value to avoid first false positive


            // Se il valore di confidenza è alto, segnala la corrispondenza
            //Console.WriteLine($"DEBUG [Webcam ID: {config.Id}] Pattern '{pattern.Description}' trovato con confidenza {maxValues[0]:F2}");
            
            //if (maxValues[0] >= 0.4)
            //{
            ////    Console.WriteLine($"PASSED [Webcam ID: {config.Id}] Pattern '{pattern.Description}' trovato con confidenza {maxValues[0]:F2}");
            //    var rect = new Rectangle(maxLocations[0], template.Size);
            //    CvInvoke.Rectangle(mat, rect, new MCvScalar(0, 255, 0), 3);
            //}
            
        //CvInvoke.Imshow($"Webcam {config.Id}", mat);

        c++;    
        }


        */



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
        double threshold = 0.6;
        foreach (Confidence confidence in confidences)
        {
            string selector = " ";
            if (confidence.isMax)
                selector = ">";

            string choosen = " ";
            if (confidence.ConfPercentage > threshold)
                choosen = "*";


            Console.WriteLine("       " + choosen + selector + confidence.Patternconfig.Description + " with confidence " + confidence.ConfPercentage);


            // chiamo l'hook e aggiorno la hashtable
            /*
            if (confidence.isMax && confidence.ConfPercentage > threshold)
            {
                if (((Confidence)hook_sent[config.Id]).hook_sent_timestamp)
                    hook_sent[config.Id] = confidence;
                confidence.hook_sent_timestamp = DateTime.Now;
            }*/



            if (confidence.isMax && confidence.ConfPercentage > threshold)
            {
                Console.WriteLine($"Triggering actions for pattern '{confidence.Patternconfig.Description}'");

                // Esegui le azioni nei hooks in un thread parallelo
                Thread hookThread = new Thread(() => ExecuteHooks(confidence.Patternconfig.Hooks, confidence.Patternconfig.Description));
                hookThread.Start();
            }

        }
        Console.SetCursorPosition(0, Console.CursorTop - confidences.Count() - 1);





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

    static void ExecuteHooks(Hook[] hooks, string qrCodeData)
    {
        foreach (var hook in hooks)
        {
            switch (hook.Type.ToLower())
            {
                case "https":
                    Console.WriteLine($"Sending https " + hook.Payload);
                    SendDataToServer(hook.Endpoint, qrCodeData, hook.Username, hook.Password, hook.Payload);
                    break;
                case "mqtt":
                    Uri uri = new Uri(hook.Endpoint);
                    Console.WriteLine($"Sending mqtts " + hook.Payload);
                    MqttPublisher.PublishMqttsMessageAsync(uri.Host, uri.Port, "CV_recognizer", hook.Username, hook.Password, hook.Topic, hook.Payload);
                    break;
                case "sleep":
                    int delay = int.Parse(hook.Payload);
                    Console.WriteLine($"Sleeping for {delay} ms");
                    Thread.Sleep(delay);
                    break;
                default:
                    Console.WriteLine($"Unknown hook type: {hook.Type}");
                    break;
            }
        }
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

    static async Task SendDataToServer(string url, string qrCodeData, string username, string password, string payloadTemplate)
    {
        using var client = new HttpClient();
        var payload = payloadTemplate.Replace("{qrCodeData}", qrCodeData);

        var request = new HttpRequestMessage(HttpMethod.Post, url)
        {
            Content = new StringContent(payload, Encoding.UTF8, "application/json")
        };

        if (!string.IsNullOrEmpty(username) && !string.IsNullOrEmpty(password))
        {
            var byteArray = Encoding.ASCII.GetBytes($"{username}:{password}");
            client.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Basic", Convert.ToBase64String(byteArray));
        }

        try
        {
            var response = await client.SendAsync(request);
            response.EnsureSuccessStatusCode();
            Console.WriteLine("INFO: HTTPS Hook executed successfully.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ERROR: HTTPS Hook failed: {ex.Message}");
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

public class Hook
{
    public string Type { get; set; }        // Tipo: html, https, mqtt, mqtts, sleep
    public string Endpoint { get; set; }    // L'URL completo del protocollo
    public string Username { get; set; }    // Username (opzionale, usato per MQTT o HTTPS)
    public string Password { get; set; }    // Password (opzionale, usato per MQTT o HTTPS)
    public string Payload { get; set; }     // Payload da inviare con POST o MQTT topic

    public string Topic { get; set; }     // Topic
}

public class PatternConfig
{
    public int Id { get; set; }
    public string Description { get; set; }
    public string Path { get; set; }

    public Hook[] Hooks { get; set; }
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

