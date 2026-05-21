unit uMainForm;

interface

uses
  System.SysUtils,
  System.Classes,
  System.JSON,
  System.Net.HttpClient,
  System.Net.Mime,

  FMX.Types,
  FMX.Controls,
  FMX.Forms,
  FMX.Graphics,
  FMX.StdCtrls,
  FMX.Objects,
  FMX.Layouts,
  FMX.Media;

const
  API_URL = 'http://192.168.0.10:8000/classify';

type
  TFormMain = class(TForm)
    CameraComponent1: TCameraComponent;
    imgPreview: TImage;
    TimerCapture: TTimer;

    lblFoodName: TLabel;
    lblConfValue: TLabel;
    lblDetect: TLabel;

    rectFill: TRectangle;

    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);

    procedure TimerCaptureTimer(Sender: TObject);

  private
    FProcessing: Boolean;

    procedure StartCamera;
    procedure CaptureAndClassify;

    procedure MostrarResultado(
      const Classe,
      Item: string;
      Confianca: Double
    );

    procedure MostrarErro(
      const Msg: string
    );

  public
  end;

var
  FormMain: TFormMain;

implementation

{$R *.fmx}

procedure TFormMain.FormCreate(Sender: TObject);
begin
  FProcessing := False;

  StartCamera;

  TimerCapture.Interval := 1500;
  TimerCapture.Enabled := True;
end;

procedure TFormMain.FormDestroy(Sender: TObject);
begin
  CameraComponent1.Active := False;
end;

procedure TFormMain.StartCamera;
begin
  CameraComponent1.Kind := TCameraKind.BackCamera;

  CameraComponent1.FocusMode :=
    TFocusMode.ContinuousAutoFocus;

  CameraComponent1.Active := True;
end;

procedure TFormMain.TimerCaptureTimer(Sender: TObject);
begin
  if FProcessing then
    Exit;

  CaptureAndClassify;
end;

procedure TFormMain.CaptureAndClassify;
var
  Bitmap: TBitmap;
  Stream: TMemoryStream;

  Http: THTTPClient;
  FormData: TMultipartFormData;
  Resp: IHTTPResponse;

  JSON: TJSONObject;

  Classe: string;
  Item: string;
  Confianca: Double;
begin
  FProcessing := True;

  lblDetect.Text := 'Processando...';

  Bitmap := TBitmap.Create;
  Stream := TMemoryStream.Create;

  Http := THTTPClient.Create;
  FormData := TMultipartFormData.Create;

  try
    CameraComponent1.SampleBufferToBitmap(
      Bitmap,
      True
    );

    imgPreview.Bitmap.Assign(Bitmap);

    Bitmap.SaveToStream(Stream);

    Stream.Position := 0;

    FormData.AddStream(
      'file',
      Stream,
      'frame.jpg',
      'image/jpeg'
    );

    Resp := Http.Post(
      API_URL,
      FormData
    );

    if Resp.StatusCode = 200 then
    begin
      JSON := TJSONObject.ParseJSONValue(
        Resp.ContentAsString
      ) as TJSONObject;

      try
        Classe :=
          JSON.GetValue<string>(
            'classe',
            ''
          );

        Item :=
          JSON.GetValue<string>(
            'item',
            ''
          );

        Confianca :=
          JSON.GetValue<Double>(
            'confianca',
            0
          );

        MostrarResultado(
          Classe,
          Item,
          Confianca
        );

      finally
        JSON.Free;
      end;
    end
    else
    begin
      MostrarErro(
        'Erro HTTP'
      );
    end;

  except
    on E: Exception do
      MostrarErro(E.Message);
  end;

  FormData.Free;
  Http.Free;
  Bitmap.Free;

  FProcessing := False;
end;

procedure TFormMain.MostrarResultado(
  const Classe,
  Item: string;
  Confianca: Double
);
var
  Pct: Integer;
begin
  if Classe = '' then
  begin
    MostrarErro(
      'Nenhum prato detectado'
    );

    Exit;
  end;

  if Item <> '' then
    lblFoodName.Text := Item
  else
    lblFoodName.Text := Classe;

  Pct := Round(
    Confianca * 100
  );

  lblConfValue.Text :=
    Pct.ToString + '%';

  rectFill.Width :=
    220 * Confianca;

  lblDetect.Text :=
    'Prato identificado';
end;

procedure TFormMain.MostrarErro(
  const Msg: string
);
begin
  lblFoodName.Text :=
    'Nenhum prato';

  lblConfValue.Text :=
    '--';

  rectFill.Width := 0;

  lblDetect.Text := Msg;
end;

end.
