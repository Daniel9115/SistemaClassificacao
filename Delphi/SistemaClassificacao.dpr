program SistemaClassificacao;

uses
  Vcl.Forms,
  uMainForm in 'uMainForm.pas' {Form1};

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TFormMain, FormMain);
  Application.Run;
end.
