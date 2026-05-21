unit uMainForm;

interface

uses
  Winapi.Windows, Winapi.Messages,
  System.SysUtils, System.Classes, System.JSON,
  System.Net.HttpClient, System.Net.Mime,
  Vcl.Graphics, Vcl.Controls, Vcl.Forms,
  Vcl.Dialogs, Vcl.StdCtrls, Vcl.ExtCtrls,
  Vcl.Imaging.jpeg;

const
  API_URL = 'http://localhost:8000/classify';

type
  TFormMain = class(TForm)

    // COMPONENTES DO DFM
    pnlHeader: TPanel;
    lblTitulo: TLabel;
    lblSubtitulo: TLabel;
    shpOnline: TShape;
    lblOnline: TLabel;

    pnlCameraCard: TPanel;
    lblCamera: TLabel;
    lblTempo: TLabel;
    imgPreview: TImage;

    pnlSidebar: TPanel;

    pnlResultado: TPanel;
    lblResultadoTitulo: TLabel;
    lblFoodName: TLabel;
    lblConfiancaTxt: TLabel;
    lblConfValue: TLabel;
    pnlTrack: TPanel;
    pnlFill: TPanel;

    pnlStatus: TPanel;
    lblStatusTitulo: TLabel;
    pnlDetect: TPanel;
    lblDetect: TLabel;

    pnlInfo: TPanel;
    lblInfoTitulo: TLabel;
    lblApiTxt: TLabel;
    lblApi: TLabel;
    lblUpdateTxt: TLabel;
    lblUpdate: TLabel;
    lblModeloTxt: TLabel;
    lblModelo: TLabel;

    dlgOpen: TOpenDialog;
    Panel1: TPanel;
    btnSelecionar: TButton;

    procedure FormCreate(Sender: TObject);
    procedure btnSelecionarClick(Sender: TObject);

  private
    procedure ClassificarImagem(const FileName: string);
    procedure MostrarErro(const Msg: string);
    procedure MostrarResultado(
      const Classe,
      Item: string;
      Confianca: Double
    );
  end;

var
  FormMain: TFormMain;

implementation

{$R *.dfm}

procedure TFormMain.FormCreate(Sender: TObject);
var
  Http: THTTPClient;
  Resp: IHTTPResponse;
begin
  dlgOpen.Filter :=
    'Imagens|*.jpg;*.jpeg;*.png;*.bmp';

  lblFoodName.Caption := 'Nenhum prato';
  lblConfValue.Caption := '--';
  lblDetect.Caption := 'Aguardando detecção';

  pnlFill.Width := 0;

  // STATUS INICIAL
  lblOnline.Caption := 'Verificando...';
  lblOnline.Font.Color := clGray;

  shpOnline.Brush.Color := clSilver;
  shpOnline.Pen.Color := clSilver;

  // TESTA API AO ABRIR O PROGRAMA
  Http := THTTPClient.Create;

  try
    try
      Resp := Http.Get('http://localhost:8000/status');

      if Resp.StatusCode = 200 then
      begin
        lblOnline.Caption := 'Online';
        lblOnline.Font.Color := clGreen;

        shpOnline.Brush.Color := clLime;
        shpOnline.Pen.Color := clLime;
      end
      else
      begin
        lblOnline.Caption := 'Offline';
        lblOnline.Font.Color := clRed;

        shpOnline.Brush.Color := clRed;
        shpOnline.Pen.Color := clRed;
      end;

    except
      lblOnline.Caption := 'Offline';
      lblOnline.Font.Color := clRed;

      shpOnline.Brush.Color := clRed;
      shpOnline.Pen.Color := clRed;
    end;

  finally
    Http.Free;
  end;
end;
procedure TFormMain.ClassificarImagem(
  const FileName: string
);
var
  Http: THTTPClient;
  FormData: TMultipartFormData;
  Resp: IHTTPResponse;
  JSON: TJSONObject;

  Classe: string;
  Item: string;
  Confianca: Double;

  Pct: Integer;
begin
  lblDetect.Caption := 'Processando...';

  Http := THTTPClient.Create;
  FormData := TMultipartFormData.Create;

  try
    FormData.AddFile(
      'file',
      FileName,
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
        'Erro HTTP: ' +
        Resp.StatusCode.ToString
      );
    end;

  except
    on E: Exception do
      MostrarErro(E.Message);
  end;

  FormData.Free;
  Http.Free;
end;

procedure TFormMain.btnSelecionarClick(Sender: TObject);
begin
  if dlgOpen.Execute then
  begin
    imgPreview.Picture.LoadFromFile(
      dlgOpen.FileName
    );

    ClassificarImagem(
      dlgOpen.FileName
    );
  end;
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
    MostrarErro('Nenhum prato detectado');
    Exit;
  end;

  if Item <> '' then
    lblFoodName.Caption := Item
  else
    lblFoodName.Caption := Classe;

  Pct := Round(
    Confianca * 100
  );

  lblConfValue.Caption :=
    IntToStr(Pct) + '%';

  pnlFill.Width :=
    Round(
      pnlTrack.Width *
      Confianca
    );

  lblDetect.Caption :=
    'Prato identificado';

  pnlDetect.Color := $00ECFDF5;

  lblDetect.Font.Color :=
    $00059669;
end;

procedure TFormMain.MostrarErro(
  const Msg: string
);
begin
  lblFoodName.Caption :=
    'Nenhum prato';

  lblConfValue.Caption :=
    '--';

  pnlFill.Width := 0;

  lblDetect.Caption := Msg;

  pnlDetect.Color :=
    $00F1F5F9;

  lblDetect.Font.Color :=
    clGray;
end;

end.
