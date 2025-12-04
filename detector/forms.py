from django import forms
class NewsForm(forms.Form):
    news_text = forms.CharField(widget=forms.Textarea, label="Paste News Headline or Article")
