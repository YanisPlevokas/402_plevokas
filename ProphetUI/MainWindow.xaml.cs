using System.Windows;


namespace ProphetUI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private readonly ProphetHelper prophetModel = new();
        public MainWindow()
        {
            InitializeComponent();
            DataContext = prophetModel;
        }
        private void ButtonOpenDirectory(object sender, RoutedEventArgs e)
        {
            using (var dialog = new System.Windows.Forms.FolderBrowserDialog())
            {
                if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    prophetModel.InputPath = dialog.SelectedPath;
                }
            }
        }
        private void ButtonStartForecast(object sender, RoutedEventArgs e)
        {
            prophetModel.StartProcessing();
        }
        private void ButtonStopForecast(object sender, RoutedEventArgs e)
        {
            prophetModel.StopProcess();
        }
    }

    
}
