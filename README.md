# Cài Git, Python
Bật PowerShell, copy từng dòng vào và bấm enter:
```PowerShell
winget install git
winget install python
```

Khởi động lại máy

# Tải code
Bật PowerShell, tiếp tục copy từng dòng vào và bấm enter:
```PowerShell
git clone https://github.com/ooker777/reddit-trend-analysis/
cd reddit-trend-analysis
```

# Cài các thư viện cần thiết
```PowerShell
pip install praw pandas numpy datetime
```

# Chạy code
```PowerShell
python hottest_reddit_posts.py
```