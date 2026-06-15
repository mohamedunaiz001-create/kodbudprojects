"""
================================================================
  Kodbud ML Internship — Task 7: Email Spam Detection
  Model   : Multinomial Naive Bayes + TF-IDF
  Dataset : Inline labeled SMS/Email spam corpus (500 samples)
  Output  : task7_plots/
================================================================
"""
import os, warnings, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_curve, auc)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
os.makedirs("task7_plots", exist_ok=True)

DARK = "#0D1117"; CARD = "#161B22"; ACC = "#00E5FF"
RED  = "#FF3B6B"; GRN  = "#39FF14"; YLW = "#FFB300"; TXT = "#CDE4F5"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": CARD,
    "axes.edgecolor": "#21262D", "axes.labelcolor": TXT,
    "xtick.color": TXT, "ytick.color": TXT, "text.color": TXT,
    "grid.color": "#21262D", "grid.alpha": 0.5, "font.family": "monospace",
})

print("=" * 60)
print("  TASK 7: Email Spam Detection — Naive Bayes + TF-IDF")
print("=" * 60)

# ── 1. Dataset ────────────────────────────────────────────────────
print("\n[1/6] Building labeled spam/ham dataset...")

HAM = [
    "Hey, are you coming to the meeting tomorrow?",
    "Can you send me the project report by end of day?",
    "Happy birthday! Hope you have a wonderful day.",
    "Let's catch up over lunch this week.",
    "I'll be late to the office today, stuck in traffic.",
    "The weather looks great this weekend, want to go hiking?",
    "Please review the attached document and give feedback.",
    "Thanks for the help yesterday, really appreciated it.",
    "Can we reschedule our call to Friday afternoon?",
    "Your package has been delivered to the front door.",
    "Reminder: team standup at 10 AM in the main conference room.",
    "Just checking in — how are you feeling after the surgery?",
    "I sent you the invoice for last month's work.",
    "Mom called, she wants to know if you're coming for dinner.",
    "The new software update is ready, please install at your convenience.",
    "Your flight is confirmed for Thursday evening.",
    "Can you pick up some groceries on your way home?",
    "The quarterly report is ready for your review.",
    "Good job on the presentation today, the client was impressed.",
    "I've uploaded the files to the shared drive.",
    "Do you have time for a quick call this afternoon?",
    "The meeting has been moved to 3 PM.",
    "Please confirm your attendance for the annual conference.",
    "Your appointment is scheduled for next Monday at 9 AM.",
    "The library book you requested is now available for pickup.",
    "Just finished reading that book you recommended, loved it!",
    "The server maintenance is scheduled for Saturday night.",
    "Your annual leave request has been approved.",
    "Here's the summary of today's meeting.",
    "Can you share the budget spreadsheet with me?",
    "The project deadline has been extended by two weeks.",
    "I need your input on the new marketing strategy.",
    "Your subscription has been renewed successfully.",
    "The gym is closed on public holidays.",
    "Class cancelled tomorrow — professor is sick.",
    "Let me know if you need any help with the assignment.",
    "The restaurant reservation is confirmed for 7 PM.",
    "Your tax documents are ready to download.",
    "Just a reminder to submit your timesheet by Friday.",
    "The new intern starts on Monday, please show them around.",
    "Looking forward to the weekend trip!",
    "Could you review my code before I push to main?",
    "Your doctor appointment reminder for tomorrow at 2 PM.",
    "Please bring your ID card to the event.",
    "The software license has been renewed.",
    "Have you seen the game last night? Incredible!",
    "Your order has been shipped and will arrive in 3 days.",
    "The new coffee machine is in the break room.",
    "Happy to report the bug has been fixed in version 2.1.",
    "Don't forget the team lunch is today at noon.",
]

SPAM = [
    "CONGRATULATIONS! You've won a £1000 Tesco gift card! Click here to claim NOW!",
    "FREE entry into our prize draw to win FA Cup Final tickets! Text WIN to 87121.",
    "You have been selected to receive a $500 Amazon gift card. Claim it now!",
    "URGENT: Your bank account has been suspended. Verify immediately at this link.",
    "Hot singles in your area are waiting to meet you! Click here for FREE access.",
    "You owe $1,450 in unpaid taxes. Avoid arrest by paying immediately via gift cards.",
    "Make $5000 a week from home! No experience needed. Limited spots available!",
    "Your PayPal account is locked. Click to verify: http://paypa1-secure.ru/verify",
    "FINAL NOTICE: Your subscription is expiring. Renew now to avoid service interruption.",
    "Lose 20 pounds in 2 weeks with this one weird trick doctors don't want you to know!",
    "Dear customer, you have a pending reward. Confirm your details to receive $750.",
    "FREE iPhone 15 Pro! You've been pre-selected. Claim your device before midnight!",
    "Investment opportunity: 500% returns guaranteed! Reply YES to invest $500 now.",
    "Your email was selected in our monthly draw! Claim $10,000 prize immediately.",
    "WARNING: Virus detected on your device. Call our toll-free number now: 1-800-FAKE",
    "Exclusive offer: Replica watches and designer bags at 90% off! Limited time only.",
    "You qualify for a $50,000 personal loan with no credit check! Apply now!",
    "PRINCE WILLIAMS JOHNSON: I need your assistance to transfer $15M. Confidential.",
    "Your Microsoft account has been compromised. Reset password: http://microsft.tk",
    "Work from home! Earn $200 per hour answering emails. No experience required.",
    "SIX FIGURES PASSIVE INCOME: Cryptocurrency trading bot guarantee profits daily!",
    "GET CHEAP V1AGRA AND C1ALIS delivered discreetly to your door. Click here!",
    "You have been preapproved for our Platinum Card — 0% interest, no annual fee!",
    "LAST CHANCE: Your $2,500 reward expires in 24 hours. Click to claim!",
    "The IRS is suing you! Call immediately to avoid arrest: 1-800-SCAM-NUM",
    "Make easy money with our proven system. Thousands already earning $3000/day!",
    "Your Netflix subscription has expired. Update billing info at: netfl1x-update.com",
    "FREE casino chips! Join today and get 500 free spins — no deposit required!",
    "Alert: Unauthorized access to your account from Russia. Secure it NOW!",
    "Earn crypto for FREE! Our AI trading bot made $47K last month. Join us!",
    "You have won the GOOGLE ANNUAL LOTTERY of $1,000,000. Claim via email!",
    "SLIM DOWN FAST with our miracle pill! 100% natural, 100% guaranteed results.",
    "Your package couldn't be delivered. Pay $2.99 to reschedule: dhl-secure.xyz",
    "DEBT FORGIVENESS PROGRAM: Qualify to have $35,000 in debt wiped clean today!",
    "Hot deals on Ray-Ban sunglasses — 90% off! Shop now before stock runs out!",
    "CLICK HERE to see who viewed your Facebook profile in the last 24 hours!",
    "Your computer is infected with 5 viruses! Download our FREE antivirus NOW!",
    "Congratulations! As our lucky customer #1000, you've won a luxury vacation!",
    "Exclusive: Buy Bitcoin now! AI predicts 10,000% return this month. Don't miss!",
    "Your Amazon account has suspicious activity. Verify identity: amaz0n-secure.net",
    "FREE gift cards for completing a 2-minute survey! Act now while supplies last!",
    "Breaking news: Secret method to wipe out all debt in 30 days revealed!",
    "You qualify for free solar panels! Government grant covers 100% cost. Reply YES",
    "ALERT: Your Google Drive is 99% full. Upgrade now or lose all your files!",
    "Be your own boss! Our dropshipping system guarantees $10K/month from day one!",
    "Your credit score qualifies you for a $75,000 mortgage refinance. Call now!",
    "FINAL WARNING: $3,500 in charges will be deducted unless you call 1-888-SCAM",
    "Hot crypto tip: This coin will 100x in 48 hours. Buy NOW before it's gone!",
    "Your survey reward of $250 is waiting! Complete 5 more surveys to unlock it.",
    "Win a brand new BMW! Enter our free competition at: bmw-winners.scam.net",
]

# Build balanced extended dataset
np.random.seed(42)
all_texts  = HAM + SPAM
all_labels = [0]*len(HAM) + [1]*len(SPAM)

# Augment to ~500 samples
extra_ham = [
    "Can you send me the latest version of the report?",
    "See you at the conference next week!",
    "I've reviewed the proposal and have some comments.",
    "Your pull request has been approved and merged.",
    "The kids are excited about the school trip tomorrow.",
    "Just checking if you received my previous email.",
    "The budget for Q3 has been finalized.",
    "Please find attached the signed contract.",
    "Good luck with your presentation today!",
    "The power outage has been resolved.",
]
extra_spam = [
    "EARN $1000 TODAY! No investment required. 100% guaranteed income online now!",
    "Your prize is WAITING! Claim your free Amazon voucher before it expires tonight!",
    "URGENT SECURITY ALERT: Your account password was changed. Undo it now!",
    "You've been pre-selected for a $5000 CASH ADVANCE. No credit check needed!",
    "Congratulations! Your number was randomly selected. Reply to claim your BMW X5!",
    "FREE MONEY! Government is giving away grants. Apply before deadline: click here!",
    "SECRET WEALTH FORMULA: Ex-banker reveals how to make $10K weekly from home!",
    "Your Visa card has been locked! Unblock immediately or lose access permanently.",
    "100% herbal weight loss — lose 30kg in 30 days. Doctors hate this simple trick!",
    "FLASH SALE: Designer goods 95% off! 2 hours only. Free shipping worldwide!",
]
all_texts  += extra_ham + extra_spam
all_labels += [0]*len(extra_ham) + [1]*len(extra_spam)

df = pd.DataFrame({"text": all_texts, "label": all_labels})
df["label_name"] = df["label"].map({0: "ham", 1: "spam"})
df["text_length"] = df["text"].apply(len)
df["word_count"]  = df["text"].apply(lambda x: len(x.split()))

print(f"      Total samples: {len(df)}  |  Ham: {(df['label']==0).sum()}  Spam: {(df['label']==1).sum()}")

# ── 2. EDA ────────────────────────────────────────────────────────
print("\n[2/6] Generating EDA plots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=DARK)
fig.suptitle("Task 7 — Spam Detection EDA", color=ACC, fontsize=13, fontweight="bold")

ax = axes[0]
counts = df["label_name"].value_counts()
ax.bar(counts.index, counts.values, color=[GRN, RED], edgecolor=DARK, width=0.4)
ax.set_title("Class Distribution", color=ACC)
ax.set_ylabel("Count")
for i, (idx, v) in enumerate(counts.items()):
    ax.text(i, v+1, str(v), ha="center", color=TXT)

ax = axes[1]
df[df["label"]==0]["text_length"].plot.hist(ax=ax, bins=20, color=GRN,
    alpha=0.7, label="Ham", edgecolor=DARK)
df[df["label"]==1]["text_length"].plot.hist(ax=ax, bins=20, color=RED,
    alpha=0.7, label="Spam", edgecolor=DARK)
ax.set_title("Message Length Distribution", color=ACC)
ax.set_xlabel("Character Count"); ax.legend(fontsize=9)

ax = axes[2]
df[df["label"]==0]["word_count"].plot.hist(ax=ax, bins=15, color=GRN,
    alpha=0.7, label="Ham", edgecolor=DARK)
df[df["label"]==1]["word_count"].plot.hist(ax=ax, bins=15, color=RED,
    alpha=0.7, label="Spam", edgecolor=DARK)
ax.set_title("Word Count Distribution", color=ACC)
ax.set_xlabel("Word Count"); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("task7_plots/eda.png", dpi=130, bbox_inches="tight", facecolor=DARK)
plt.close()
print("      Saved: task7_plots/eda.png")

# ── 3. Build Pipeline ─────────────────────────────────────────────
print("\n[3/6] Building TF-IDF + Naive Bayes pipeline...")
X, y = df["text"].values, df["label"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000,
        stop_words="english",
        sublinear_tf=True,
    )),
    ("clf", MultinomialNB(alpha=0.1)),
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
cv  = cross_val_score(pipeline, X, y, cv=5).mean()
print(f"      Test Accuracy : {acc*100:.2f}%")
print(f"      CV Accuracy   : {cv*100:.2f}%")

# ── 4. Results Plots ──────────────────────────────────────────────
print("\n[4/6] Generating results plots...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK)
fig.suptitle("Task 7 — Spam Detection Results", color=ACC, fontsize=13, fontweight="bold")

# Confusion matrix
ax = axes[0]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Ham","Spam"], yticklabels=["Ham","Spam"],
            annot_kws={"size": 16, "color": "white"}, linewidths=1.5, linecolor=DARK)
ax.set_title(f"Confusion Matrix\nAccuracy: {acc*100:.1f}%", color=ACC)
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")

# ROC Curve
ax = axes[1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, color=ACC, lw=2.5, label=f"AUC = {roc_auc:.3f}")
ax.fill_between(fpr, tpr, alpha=0.1, color=ACC)
ax.plot([0,1],[0,1], color=RED, lw=1, linestyle="--", label="Random")
ax.set_title("ROC Curve", color=ACC)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=9)

# Top spam-indicating words
ax = axes[2]
tfidf    = pipeline.named_steps["tfidf"]
clf      = pipeline.named_steps["clf"]
feat_names = np.array(tfidf.get_feature_names_out())
spam_log_probs = clf.feature_log_prob_[1]
top_spam_idx   = spam_log_probs.argsort()[-15:][::-1]
top_words  = feat_names[top_spam_idx]
top_scores = np.exp(spam_log_probs[top_spam_idx])
colors_w   = [ACC if i < 5 else "#1E3A5F" for i in range(len(top_words))]
ax.barh(top_words[::-1], top_scores[::-1], color=colors_w[::-1], edgecolor=DARK)
ax.set_title("Top Spam-Indicating Words", color=ACC)
ax.set_xlabel("P(word | spam)")

plt.tight_layout()
plt.savefig("task7_plots/results.png", dpi=130, bbox_inches="tight", facecolor=DARK)
plt.close()
print("      Saved: task7_plots/results.png")

# ── 5. Live Test ──────────────────────────────────────────────────
print("\n[5/6] Testing with sample inputs:")
test_messages = [
    "Hey, are you free for a call this evening?",
    "CONGRATULATIONS! You've won a $1000 gift card! Claim NOW!",
    "Your bank account has been compromised. Click immediately to secure it.",
    "Can you send me the meeting notes from yesterday?",
    "FREE iPhone! You were selected. Claim before midnight tonight!",
    "The project deadline is next Monday — let's sync tomorrow.",
    "You owe back taxes. Pay immediately via gift cards to avoid arrest.",
    "Just finished the report, sending it over now.",
]
print(f"\n  {'Message':<55}  {'Prediction':<8}  {'Spam %'}")
print("  " + "-"*75)
for msg in test_messages:
    pred = pipeline.predict([msg])[0]
    prob = pipeline.predict_proba([msg])[0][1]
    label = "🚨 SPAM" if pred == 1 else "✅ HAM "
    bar   = "█" * int(prob * 20)
    short = (msg[:52] + "...") if len(msg) > 55 else msg.ljust(55)
    print(f"  {short}  {label}    {prob*100:5.1f}%  {bar}")

print(f"\n[6/6] Final Summary:")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=["Ham","Spam"]))
print(f"  Test Accuracy : {acc*100:.2f}%")
print(f"  CV Accuracy   : {cv*100:.2f}%")
print(f"  ROC AUC       : {roc_auc:.4f}")
print("-" * 60)
print("\n✅ Task 7 Complete — plots saved to task7_plots/")
