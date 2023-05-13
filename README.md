## Bật PowerShell

Bấm chuột phải vào logo Windows trên thanh taskbar (hoặc bấm <kbd>Win + X</kbd>)
![](https://thegeekpage.com/wp-content/uploads/2022/04/press-windows-and-x-keys_11zon.png)

Chọn Terminal (hoặc Command Prompt nếu không có Terminal)

## Cài Git, Python

Sau khi bật được PowerShell, copy từng dòng sau vào và bấm enter:

```PowerShell
winget install git.git -source
winget install python
```

## Tải code

Tắt PowerShell đi và bật lại. Tiếp tục copy từng dòng vào và bấm enter:

```PowerShell
git clone https://github.com/ooker777/reddit-trend-analysis/
cd reddit-trend-analysis
```

## Cài các thư viện cần thiết

```PowerShell
pip install praw pandas numpy datetime
```

## Chạy code

```PowerShell
python hottest_reddit_posts.py
```
