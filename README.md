# ARIAKE OCTA Stack Registration Tool

Retinal OCTA 画像スタックのレジストレーションと平均化のためのデスクトップアプリです。ARIAKE ImageJ マクロの Python 実装をベースに、単体実行と操作性を重視しています。

## 機能

- **ImageJ との整合**: 前処理、4 倍拡大、CLAHE 最適化などを ImageJ の挙動に合わせて再現。
- **参照ベースのアライメント**: 「平均スタック（画像 5）」方式による Affine レジストレーション。
- **信頼度スコアと自動リファイン + 手動補正**: 各キャプチャの自動アライメント信頼度を算出し、
  低いものには**特徴点ベース（ORB + RANSAC）の自動リファイン**を試行。それでも閾値に届かない
  ものだけをユーザーが選択して、対応点（corresponding points）を手動指定して補正できます。
- **データ検証**: 患者フォルダ間でファイル順序と整合性を自動確認。
- **UI**: Flet によるモダンなクロスプラットフォーム UI。

## ワークフロー

1. **自動レジストレーション**（`Run Registration`）: 従来どおり全自動で処理します。
2. **確認と手動補正**（`Review & Correct`）: 元画像の品質が低く自動アライメントが
   うまくいかないキャプチャ向けのワークフローです。

### 手動対応点補正（Review & Correct）

各 Visit について「画像 5」参照スタックと自動アライメント行列を計算し、キャプチャごとの
**信頼度スコア**（参照との正規化相互相関）を表示します。しきい値（既定 `0.80`）を下回る
キャプチャには、まず**特徴点ベースの自動リファイン**（ORB 特徴 + RANSAC による部分アフィン）
を試み、同じ相関指標で改善した場合のみ自動採用します。それでもしきい値未満のキャプチャは
赤で警告表示され、手動補正の対象になります。

> **補足（RetinaRegNet の検討）**: 拡散モデル特徴を用いる
> [RetinaRegNet](https://github.com/mirthAI/RetinaRegNet) も候補として調査しましたが、
> Stable Diffusion 2-1 + DINO 系の重量級依存（torch/diffusers/xformers、数 GB のモデル、
> **CUDA 必須**）が本ツールの軽量なデスクトップ配布（`flet build` / PyInstaller、CPU 前提）
> と適合せず、また対象タスク（大変位・低重複・マルチモーダル網膜レジストレーション）が本件の
> 「同一 OCTA の微小ずれをアフィンで平均化」とは異なるため、代わりに CPU のみで動く軽量な
> 特徴点ベース自動リファインを採用しています。詳細は Pull Request の説明を参照してください。

1. 左のリストから補正したいキャプチャを選択します。
2. 参照（キャプチャ 1・緑）と対象（マゼンタ）の 2 画像に、**同じ解剖学的特徴**を
   3 点以上、対応する順序でクリックします。
3. `Compute & Preview` で対応点から Affine 変換を推定し、重ね合わせプレビューと
   自動／手動それぞれのアライメントスコアを表示します（白く重なるほど良好）。
4. 手動結果が良ければ `Accept`、やり直すなら `Reset`、自動に戻すなら `Revert to auto`。
5. `Finalize & Save` で、補正した行列を使って全レイヤーをワープ・平均化して保存します。
   補正内容（行列・対応点・スコア）は出力先の `alignment_session.json` に記録され、
   再現・再実行に利用できます。

対応点は 3 点で厳密解、4 点以上では最小二乗（LMEDS でロバスト推定）になります。

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
