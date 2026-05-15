# ARIAKE OCTA Stack Registration Tool

Retinal OCTA 画像スタックのレジストレーションと平均化のためのデスクトップアプリです。ARIAKE ImageJ マクロの Python 実装をベースに、単体実行と操作性を重視しています。

## 機能

- **ImageJ との整合**: 前処理、4 倍拡大、CLAHE 最適化などを ImageJ の挙動に合わせて再現。
- **参照ベースのアライメント**: 「平均スタック（画像 5）」方式による Affine レジストレーション。
- **データ検証**: 患者フォルダ間でファイル順序と整合性を自動確認。
- **UI**: Flet によるモダンなクロスプラットフォーム UI。

## 要件

- **Python 3.14**（推奨。プロジェクトは 3.14 前提で開発）
- 依存関係: `requirements.txt`（現行 venv に合わせ `pip freeze` でバージョン固定）

仮想環境（`venv/`、`.venv/` など）とキャッシュ類は `.gitignore` で除外されています。

## インストール

1. **リポジトリを取得**:
   ```bash
   git clone <repository-url>
   cd stack_reg
   ```

2. **仮想環境**:
   ```bash
   python3.14 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **依存関係のインストール**:
   ```bash
   pip install -r requirements.txt
   ```

## 実行

開発モードでアプリを起動:

```bash
python -m app.main
```

## デスクトップ向けパッケージ（.exe / .dmg）

`flet build` でネイティブ実行ファイルを作成できます。

### Windows (.exe)

**前提（初回のみ）**

1. 仮想環境を有効化し、依存関係をインストール済みであること（[インストール](#インストール)参照）。
2. **開発者モードを ON** にする（Flutter プラグインのシンボリックリンクに必要）。  
   `Win + I` → **システム** → **開発者向け** → **開発者モード** をオン。  
   または PowerShell で設定を開く: `start ms-settings:developers`
3. 初回ビルド時は Flutter SDK が自動ダウンロードされます（数 GB、時間がかかります）。

**ビルド**

```powershell
cd c:\Users\Y\stack_reg
.\scripts\build_windows.ps1
```

手動で実行する場合（コンソールは UTF-8 推奨）:

```powershell
chcp 65001
$env:PYTHONUTF8 = "1"
$env:FLET_CLI_NO_RICH_OUTPUT = "1"
.\venv\Scripts\flet.exe build windows --yes --no-rich-output
```

**配布用パッケージ（推奨）**

```powershell
.\scripts\build_windows.ps1
```

成果物:

| 種類 | パス |
|------|------|
| フォルダ（そのまま配布可） | `dist\ARIAKE_OCTA_Stack_Registration_v0.1.0_win64\` |
| ZIP（配布用） | `dist\ARIAKE_OCTA_Stack_Registration_v0.1.0_win64.zip` |

エンドユーザーは ZIP を解凍し、`ARIAKE_OCTA_Stack_Registration.exe` を実行します（Python 不要）。

中間ビルド成果物は `build\flutter\build\windows\x64\runner\Release\` にもあります。
`flet build` の INSTALL 段階は VC ランタイム DLL の都合でスクリプトが自動修復します。

### macOS (.dmg)

```bash
flet build macos
```

成果物は `build/` 配下に出力されます（`build/` は Git 対象外）。

## プロジェクト構成（概要）

| パス | 内容 |
|------|------|
| `app/main.py` | エントリ（`ft.run`） |
| `app/ui/` | ダッシュボード・QC ビューアなど |
| `app/core/` | 画像処理・レジストレーション・パイプライン |
| `archive/` | 開発・検証用スクリプト（アプリ本体からは未使用） |

## 開発メモ

- Flet は **v0.84+** を想定（`FilePicker` は `page.services` に追加するなど API 変更あり）。
- UI / API の細則は `.cursorrules` を参照。

---

Developed by Team Yanagi (2025/2026)
