using MQTTnet.Client;
using MQTTnet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using MQTTnet;
using MQTTnet.Client;
using System;
using System.Text;
using System.Security.Cryptography.X509Certificates;
using System.Threading.Tasks;


namespace QrTrainAutomation
{

    class MqttPublisher
    {

        public static async Task PublishMqttsMessageAsync(string brokerAddress, int port, string clientId, string username, string password, string topic, string message)
        {
            // Crea un'istanza del client MQTT
            var factory = new MqttFactory();
            var mqttClient = factory.CreateMqttClient();

            // Configura le opzioni di connessione con TLS
            var options = new MqttClientOptionsBuilder()
                .WithClientId(clientId)
                .WithTcpServer(brokerAddress, port)
                .WithCredentials(username, password)
                .WithTls(new MqttClientOptionsBuilderTlsParameters
                {
                    UseTls = true,
                    IgnoreCertificateChainErrors = true,  // Ignora errori della catena certificati (opzionale)
                    IgnoreCertificateRevocationErrors = true,  // Ignora errori di revoca certificati (opzionale)
                    AllowUntrustedCertificates = true  // Permette certificati non fidati (opzionale)
                })
                .WithCleanSession()
                .Build();

            // Connetti al broker MQTT
            await mqttClient.ConnectAsync(options, System.Threading.CancellationToken.None);

            // Crea il messaggio MQTT
            var mqttMessage = new MqttApplicationMessageBuilder()
                .WithTopic(topic)
                .WithPayload(Encoding.UTF8.GetBytes(message))
                .WithQualityOfServiceLevel(MQTTnet.Protocol.MqttQualityOfServiceLevel.AtLeastOnce) // QoS 1
                .WithRetainFlag(false)
                .Build();

            // Pubblica il messaggio
            await mqttClient.PublishAsync(mqttMessage, System.Threading.CancellationToken.None);

            // Disconnetti il client
            await mqttClient.DisconnectAsync();
        }
    }

}

