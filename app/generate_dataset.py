"""
Synthetic support-ticket dataset generator.

Run directly to (re)create data/tickets.csv:

    python -m app.generate_dataset

How variety is created (so records are not just "template with a word swapped"):
  * Each department has ~12 scenarios (e.g. "double charge", "login problem").
  * Each scenario has several subjects, several core sentences and several
    optional detail sentences, all with random slot values (amounts, dates...).
  * Bodies are assembled from optional greeting + core + details + closing + sign-off.
  * Writing styles vary: formal, casual, terse, all-lowercase, typos, ALL-CAPS
    subjects, "Re:" / "Fwd:" prefixes.
  * Some tickets mention a side issue from another department (cross-topic noise).
  * A small share of tickets are vague on purpose. Their label is a coin flip,
    just like real life where a vague message could land in any queue.

Everything is seeded, so the same file is produced on every run.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "tickets.csv"

SEED = 42
DEPARTMENT_TARGETS = {"Billing": 440, "Technical": 500, "HR": 440, "General": 420}
VAGUE_SHARE = 0.04       # share of tickets per department that are deliberately vague
CROSS_TOPIC_SHARE = 0.12  # share of tickets that mention an issue from another team

# --------------------------------------------------------------------------
# Slot values
# --------------------------------------------------------------------------
PLANS = ["Starter", "Pro", "Business", "Enterprise", "Team", "Premium", "Basic"]
PRODUCTS = ["the dashboard", "the mobile app", "the web app", "the desktop client",
            "the reporting module", "the analytics tool", "the scheduler", "the admin console"]
DEVICES = ["iPhone 14", "Pixel 7", "Windows 11 laptop", "MacBook Air", "Android tablet",
           "Galaxy S23", "work desktop", "Chromebook"]
BROWSERS = ["Chrome", "Firefox", "Edge", "Safari"]
ERRORS = ["500", "502", "503", "504", "403", "ERR_CONNECTION_RESET", "E1042", "invalid_grant",
          "SAML_RESPONSE_INVALID", "timeout exceeded"]
ROLES = ["Data Analyst", "Support Engineer", "Product Designer", "Backend Developer",
         "Marketing Manager", "Account Executive", "QA Engineer", "HR Generalist",
         "Finance Associate", "Customer Success Manager"]
CITIES = ["Kolkata", "Pune", "Berlin", "Austin", "Toronto", "Manchester", "Singapore", "Sydney"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
FIRST_NAMES = ["Priya", "Daniel", "Aisha", "Rahul", "Maria", "Tom", "Sneha", "Chen", "Olivia",
               "Arjun", "Fatima", "Lucas", "Ananya", "Noah", "Ishaan", "Emma"]


def make_slots(rng: random.Random) -> dict:
    """Fresh random values for placeholders such as {amount} and {date}."""
    return {
        "amount": f"${rng.choice([9, 12, 19, 29, 49, 79, 99, 149, 249, 399, 899])}.{rng.choice(['00', '99', '50'])}",
        "invoice": f"INV-{rng.randint(10000, 99999)}",
        "date": rng.choice([f"{rng.choice(MONTHS)} {rng.randint(1, 28)}", "last Friday", "yesterday",
                            f"the {rng.randint(2, 28)}th", "this Monday", "two days ago"]),
        "plan": rng.choice(PLANS),
        "product": rng.choice(PRODUCTS),
        "device": rng.choice(DEVICES),
        "browser": rng.choice(BROWSERS),
        "error": rng.choice(ERRORS),
        "role": rng.choice(ROLES),
        "city": rng.choice(CITIES),
        "card": str(rng.randint(1000, 9999)),
        "n_days": str(rng.choice([2, 3, 5, 7, 10, 14, 21, 30])),
        "n": str(rng.choice([5, 10, 25, 40, 80, 150, 300])),
        "name": rng.choice(FIRST_NAMES),
    }


# --------------------------------------------------------------------------
# Scenarios.  Each scenario: subjects, core sentences, optional details.
# --------------------------------------------------------------------------
BILLING = [
    {   # duplicate charge
        "subjects": ["Charged twice for {plan}", "Duplicate charge on my card", "Double payment this month",
                     "Two identical charges of {amount}", "Why was I billed two times?"],
        "core": ["I was billed {amount} twice on {date} for the same subscription.",
                 "My bank statement shows two identical charges from your company, both {amount}.",
                 "The card ending in {card} was hit with a duplicate transaction for {plan}. I only have one account.",
                 "Why did you take two payments in one day? I need the extra one returned.",
                 "there are 2 charges for {amount} on my credit card from you guys, one of them has to be an error"],
        "detail": ["Both charges posted within a few minutes of each other.",
                   "I've attached a screenshot of my statement.",
                   "The invoice number is {invoice}.",
                   "This is the second time this has happened."],
    },
    {   # refunds
        "subjects": ["Refund request", "Please refund my last payment", "Want my money back for {plan}",
                     "Refund for unused subscription", "Where is my refund?"],
        "core": ["I'd like a full refund for the {plan} plan I purchased {n_days} days ago. I haven't used it at all.",
                 "We cancelled before the renewal date but were still billed {amount}. Please process a refund.",
                 "Requesting a refund of {amount}, the charge was made by mistake.",
                 "You approved my refund last week but nothing has reached my account yet. How long does it normally take?",
                 "I signed up for the annual option by accident instead of monthly and would like the difference refunded."],
        "detail": ["I wrote to your team earlier and was told it would be handled.",
                   "Reference: {invoice}.",
                   "I paid with the card ending {card}.",
                   "I'm within the 30 day money-back window."],
    },
    {   # invoice problems
        "subjects": ["Invoice {invoice} has wrong company name", "Need a corrected invoice", "Missing invoice for {date}",
                     "VAT number missing on invoice", "Can you resend my invoice?"],
        "core": ["The invoice we received shows the wrong billing address and our finance team can't process it.",
                 "Could you resend {invoice}? It never arrived in our accounts payable inbox.",
                 "Our tax registration number isn't on the last three invoices and we need it for filing.",
                 "The line items on {invoice} don't match what we agreed in the contract. We were quoted a lower rate.",
                 "I can't find the invoice for {date} anywhere in my account. Please send a PDF copy."],
        "detail": ["Our accountant needs this before month end.",
                   "The company name should read Northwind Traders Pvt Ltd.",
                   "Please issue a credit note and a new invoice.",
                   "We pay on 30 day terms so timing matters."],
    },
    {   # failed payment / card issues
        "subjects": ["Payment failed", "Card declined when upgrading", "Unable to update payment method",
                     "Payment method keeps getting rejected", "Billing failed, subscription at risk"],
        "core": ["My card keeps getting declined at checkout even though it works everywhere else.",
                 "I got an email saying my payment for {plan} failed and my subscription may be suspended. The card was just renewed.",
                 "I'm trying to add a new credit card on the billing page but it says the payment couldn't be processed.",
                 "The bank says they are not blocking anything, yet your system shows the payment as failed for {amount}.",
                 "Payment didn't go through on {date}. Can I pay by another method before my plan lapses?"],
        "detail": ["I've tried two different cards.",
                   "The card ending {card} is valid until 2028.",
                   "I don't want to lose access to my plan.",
                   "Is there a grace period before the account is downgraded?"],
    },
    {   # unexpected renewal / cancel billing
        "subjects": ["Cancel my subscription and stop billing", "Unexpected renewal charge", "Auto-renew charged me",
                     "Stop recurring payments", "Charged after cancellation"],
        "core": ["I was charged {amount} for an annual renewal I didn't expect. I thought auto-renew was off.",
                 "Please cancel my {plan} subscription effective today and make sure I'm not billed again.",
                 "How do I stop the monthly charge? I can't find where to turn off the recurring payment.",
                 "I cancelled in {month} but my card was charged again today.".replace("{month}", "March"),
                 "We downsized and no longer need the {plan} subscription. Please end billing from the next cycle."],
        "detail": ["I never got a reminder email before the renewal.",
                   "I'd like written confirmation that billing has stopped.",
                   "Please confirm the date of my last charge.",
                   "The renewal charge appeared on {date}."],
    },
    {   # bill went up / overcharged
        "subjects": ["Price increase on my plan", "Why did my bill go up?", "Charged more than quoted",
                     "Overcharged this month", "Extra seat fee on invoice"],
        "core": ["My monthly bill jumped to {amount} with no notice. It used to be much lower.",
                 "Your sales rep quoted me {amount} per month but the invoice says something different.",
                 "There's an extra seat charge on my bill and I never added any new users.",
                 "I'm being billed at the old rate card instead of the discounted price in our agreement.",
                 "Can you break down what the {amount} charge covers? It looks higher than usual."],
        "detail": ["Nothing changed on our side.",
                   "We've been customers for three years.",
                   "I'd like a corrected invoice if this is a mistake.",
                   "Our contract renewal was signed in January."],
    },
    {   # proration, upgrade, downgrade
        "subjects": ["Proration question on upgrade", "Downgrade billing question", "Mid-cycle plan change charge",
                     "How is the upgrade price calculated?"],
        "core": ["I upgraded from Starter to {plan} halfway through the month. How is the prorated amount worked out?",
                 "If I downgrade now, will I get a credit for the remaining days of my billing cycle?",
                 "I changed plans on {date} and the charge on my card doesn't match what the pricing page said.",
                 "Does adding more users mid-month bill me for the full month or just the days left?"],
        "detail": ["I'd like to understand before I confirm.",
                   "Our billing day is the 1st.",
                   "A short explanation or example would be great."],
    },
    {   # promo/discount
        "subjects": ["Promo code not applied", "Discount missing from invoice", "Coupon didn't work",
                     "Nonprofit discount not showing"],
        "core": ["I used the promo code SAVE20 at checkout but the full price was charged.",
                 "Our nonprofit discount was supposed to appear on {invoice} and it isn't there.",
                 "The coupon from your newsletter says 25 percent off but my invoice shows the regular amount.",
                 "I was promised two free months when I switched to annual billing. Where is the credit?"],
        "detail": ["Please apply the discount retroactively.",
                   "I have the confirmation email if you need it.",
                   "The code was valid until the end of the month."],
    },
    {   # receipts and tax documents
        "subjects": ["Need receipt for expense report", "Tax invoice request", "Receipts for the past year",
                     "GST invoice needed"],
        "core": ["Please send receipts for all payments from the last 12 months for my expense report.",
                 "I need an itemised tax invoice including GST for my accountant.",
                 "My company reimburses software costs and wants a proper receipt for {invoice}.",
                 "Can you email me a statement of all charges between January and {date}?"],
        "detail": ["PDF format is fine.",
                   "It's for our audit.",
                   "We need it by the end of the week."],
    },
    {   # billing address, currency, payment terms
        "subjects": ["Change billing address", "Can we pay by bank transfer?", "Billed in wrong currency",
                     "Update company details on invoices"],
        "core": ["We're based in India and were billed in USD. Can we be invoiced in INR instead?",
                 "Is it possible to pay by wire transfer rather than card? Our finance policy doesn't allow credit cards.",
                 "We moved offices. Please update the billing address on our account to {city}.",
                 "I'd like to switch from monthly card payments to an annual invoice with net 30 terms."],
        "detail": ["Please send the bank details.",
                   "We'll need a purchase order number on the invoice.",
                   "Exchange fees from the card are adding up."],
    },
    {   # unauthorized charges
        "subjects": ["Unauthorized charge", "Charge I don't recognize", "Suspicious payment on my statement",
                     "I never signed up but was billed"],
        "core": ["There's a charge of {amount} on my statement from your company and I don't recognise it. I never signed up.",
                 "Someone may have used my card on your site. I see a payment I didn't make on {date}.",
                 "I closed my account last year and I'm still seeing charges. This needs to be reversed.",
                 "A {amount} payment appeared and I don't know what it's for. Can you tell me which account it belongs to?"],
        "detail": ["I'm considering disputing it with my bank.",
                   "I have already blocked the card.",
                   "Please investigate and refund if it's invalid."],
    },
    {   # overdue / payment status
        "subjects": ["Overdue invoice reminder", "Past due balance but I paid", "Payment not showing on account",
                     "Account flagged for non-payment"],
        "core": ["I received a past due notice for {invoice} but I paid this last week. Please check the payment status.",
                 "Why was my account flagged as overdue? I sent a bank transfer on {date}.",
                 "My payment of {amount} isn't reflected in my balance. Here is the transaction reference from my bank.",
                 "Your collections email says I owe money, but all my invoices up to now were settled."],
        "detail": ["Please confirm the balance is zero.",
                   "I can share the bank receipt.",
                   "This is causing trouble with our procurement team."],
    },
]

TECHNICAL = [
    {   # login
        "subjects": ["Can't log in", "Login keeps failing", "Account locked after failed attempts",
                     "Password reset email not arriving", "Sign-in page just reloads"],
        "core": ["I can't sign in. It says my credentials are invalid but I just reset my password.",
                 "The password reset link never shows up in my inbox or spam folder.",
                 "My account got locked after a few failed attempts and the unlock email isn't coming.",
                 "When I click Sign in the page just refreshes and nothing happens in {browser}.",
                 "can't get into my account, keeps saying wrong password even though I'm 100% sure it's right"],
        "detail": ["I've tried clearing cookies and a private window.",
                   "It worked fine yesterday.",
                   "I'm using {browser} on a {device}.",
                   "My colleague can log in fine with the same network."],
    },
    {   # 2FA / SSO
        "subjects": ["2FA code not accepted", "SSO login loop", "SAML error after certificate change",
                     "Authenticator app codes rejected", "Single sign-on failing for the whole team"],
        "core": ["The verification code from my authenticator app is rejected every time.",
                 "Our team is redirected back to the login screen in a loop when using single sign-on.",
                 "After we rotated our SAML certificate nobody in the company can log in. We see error {error}.",
                 "I got a new phone and lost my authenticator. How do I regain access without the codes?",
                 "The SSO integration with our identity provider returns {error} after the redirect."],
        "detail": ["Time on my phone is synced.",
                   "We use Okta as our identity provider.",
                   "No configuration changes were made on your side as far as we know.",
                   "Backup codes were not saved."],
    },
    {   # crashes
        "subjects": ["App crashes on launch", "{product} keeps freezing", "Crash when opening reports",
                     "Mobile app closes by itself", "Browser tab crashes on export"],
        "core": ["{product} crashes immediately after I open it on my {device}.",
                 "Every time I click Export the page freezes and then the tab crashes.",
                 "Since the latest update the app closes by itself when I open the settings screen.",
                 "I get a white screen and then the whole {browser} window stops responding.",
                 "The app force-closes whenever I try to attach a photo."],
        "detail": ["I reinstalled it twice.",
                   "Happens every single time.",
                   "The app version is 4.{n}.",
                   "I can send the crash log if that helps."],
    },
    {   # API
        "subjects": ["API returning errors", "Rate limit errors on our integration", "Webhooks not firing",
                     "Timeouts on REST endpoint", "401 on every API call"],
        "core": ["Our integration started getting {error} responses from the /v2/orders endpoint about an hour ago.",
                 "Webhooks stopped being delivered to our endpoint yesterday. The delivery log shows timeouts.",
                 "We're hitting 429 rate limit errors even though we're far below the documented quota.",
                 "The API token we generated returns 401 unauthorized for every request.",
                 "Pagination on the list endpoint returns duplicate records when the cursor is reused."],
        "detail": ["Request ID: req_{n}8f21.",
                   "Nothing changed in our code.",
                   "Production traffic is affected.",
                   "Here's the curl command to reproduce it."],
    },
    {   # performance
        "subjects": ["Dashboard very slow", "Pages take forever to load", "Reports time out",
                     "Everything lags since the update", "Search is extremely slow"],
        "core": ["The dashboard takes more than a minute to load now. It used to be instant.",
                 "Large reports spin forever and then time out with a gateway error.",
                 "Typing in the search box lags by several seconds on {browser}.",
                 "Since this week every page in {product} is slow for everyone on our team.",
                 "Loading a project with a few thousand rows freezes the interface."],
        "detail": ["Our internet is fine, other sites are fast.",
                   "It's slowest in the mornings.",
                   "Tested on two different networks."],
    },
    {   # sync / data
        "subjects": ["Data not syncing", "Mobile and web out of sync", "Records missing after import",
                     "Changes not saving", "Duplicate entries after sync"],
        "core": ["Changes I make on my phone don't show up on the web version, even after hours.",
                 "After importing a CSV, about a third of the rows are missing without any error message.",
                 "Edits are not saved. I refresh and my changes are gone.",
                 "The sync with our calendar duplicates every event.",
                 "The app says 'last synced {date}' and refuses to update."],
        "detail": ["I tried logging out and in.",
                   "It affects only my account.",
                   "File was 2 MB, well under the limit."],
    },
    {   # outage
        "subjects": ["Service is down", "Site not loading for entire team", "Cannot access platform",
                     "Outage?", "Everything shows an error page"],
        "core": ["Nothing is loading for anyone in our office. We just see a {error} error page.",
                 "Is there an outage? The platform has been unreachable since this morning.",
                 "The system is down for all of our users and we have customers waiting.",
                 "The status page says all good but we cannot reach the application at all.",
                 "Our whole department gets a blank page with {error}. Please look into it."],
        "detail": ["It started around 9 AM.",
                   "Multiple people confirmed on different networks.",
                   "We have a client demo shortly.",
                   "Is there an ETA for a fix?"],
    },
    {   # uploads / exports / notifications
        "subjects": ["File upload fails", "Export to CSV not working", "Notification emails not being sent",
                     "Attachments won't upload", "PDF export is blank"],
        "core": ["Uploading any file larger than 5 MB fails with an unknown error.",
                 "The CSV export downloads an empty file with only the header row.",
                 "We stopped receiving notification emails for new comments a few days ago.",
                 "Exported PDFs come out blank on {browser} but fine on another browser.",
                 "Attachments show a spinner and never finish uploading."],
        "detail": ["Checked spam folders already.",
                   "File type is a normal PDF.",
                   "Started after last week's release."],
    },
    {   # UI bugs
        "subjects": ["Button doesn't respond", "Layout broken on mobile", "Dark mode text unreadable",
                     "Dropdown menu cut off", "Bug in the date picker"],
        "core": ["The Save button does nothing when I click it on {browser}.",
                 "On my {device} the menu overlaps the content and I can't scroll.",
                 "In dark mode some labels are black on a dark background so I can't read them.",
                 "The date picker always jumps back to today's date after I choose another day.",
                 "The dropdown list is cut off at the bottom of the screen and I can't select the last option."],
        "detail": ["Screenshot attached.",
                   "I can reproduce it every time.",
                   "Steps: open settings, click profile, then try to save."],
    },
    {   # install / compatibility
        "subjects": ["Installation fails on Windows 11", "Update stuck at 90%", "Not compatible with latest {browser}",
                     "Installer throws an error", "Plugin won't load after update"],
        "core": ["The installer exits with error {error} halfway through on Windows 11.",
                 "The desktop client update has been stuck at 90 percent for an hour.",
                 "After {browser} updated, the extension won't load anymore.",
                 "I upgraded to the newest version and now my custom plugin throws errors on start.",
                 "Setup fails on my {device} saying a required component is missing."],
        "detail": ["I ran it as administrator.",
                   "Antivirus is disabled.",
                   "Full error log is below."],
    },
    {   # integrations
        "subjects": ["Slack integration disconnected", "Calendar sync not working", "Zapier connection error",
                     "OAuth connection keeps expiring"],
        "core": ["Our Slack integration disconnected overnight and reconnecting gives {error}.",
                 "Google Calendar events stopped syncing about two days ago.",
                 "The Zapier trigger returns an authentication error even after I re-authorised.",
                 "The OAuth connection to our CRM expires every day and has to be redone."],
        "detail": ["We removed and re-added the app.",
                   "The admin account was used for setup.",
                   "Other integrations work normally."],
    },
    {   # security alert
        "subjects": ["Suspicious login alert", "Account accessed from unknown device", "Possible account compromise"],
        "core": ["I got an alert about a login from another country. I'm worried someone got into my account.",
                 "My session list shows a device I don't own. How can I sign everyone out?",
                 "Someone changed my profile settings and I didn't do it. I need the access logs."],
        "detail": ["I've already changed my password.",
                   "Please look at the audit log for the last week."],
    },
]

HR = [
    {   # application status
        "subjects": ["Status of my job application", "Application for {role}", "Haven't heard back after interview",
                     "Update on my candidacy", "Following up on {role} application"],
        "core": ["I applied for the {role} position {n_days} days ago and haven't received any update. Where does my application stand?",
                 "I had a final round interview last week for {role}. When can I expect a decision?",
                 "I'd like to withdraw my application for the {role} role. I accepted another offer.",
                 "Could you confirm you received my résumé for the open {role} vacancy?",
                 "I got a rejection email but no feedback. Is it possible to know what I could improve?"],
        "detail": ["I applied through the careers page.",
                   "My application reference is APP-{n}41.",
                   "I'm happy to provide references if needed.",
                   "I have another offer deadline coming up."],
    },
    {   # interview scheduling
        "subjects": ["Reschedule my interview", "Interview time conflict", "Which time zone is my interview in?",
                     "Interview confirmation needed"],
        "core": ["I can't make my interview on {date}. Is there another slot available?",
                 "Please confirm the time zone for my video interview. The invite didn't mention it.",
                 "I haven't received the interview link for the {role} panel. Could you resend it?",
                 "Could the technical round be moved to the afternoon? I have a prior commitment."],
        "detail": ["I'm flexible on the rest of the week.",
                   "Sorry for the inconvenience.",
                   "Please let me know who I should contact on the day."],
    },
    {   # leave
        "subjects": ["Leave request", "Sick leave policy question", "Parental leave details",
                     "Vacation days balance", "Request for time off"],
        "core": ["I'd like to apply for {n_days} days of annual leave starting {date}. Who needs to approve it?",
                 "How many paid sick days do I get per year, and do I need a doctor's note?",
                 "We are expecting a baby in the spring. Can you explain the parental leave policy and how to apply?",
                 "My leave balance looks wrong in the system. It shows fewer days than I've accrued.",
                 "I need to take emergency leave for a family matter starting tomorrow. What's the procedure?"],
        "detail": ["My manager is already aware.",
                   "I've carried over some days from last year.",
                   "I'm on the engineering team.",
                   "Is unpaid leave an option if I run out?"],
    },
    {   # payroll
        "subjects": ["Salary not credited", "Payslip missing", "Wrong deduction on payslip",
                     "Update my bank details for payroll", "Overtime missing from pay"],
        "core": ["My salary for this month hasn't been credited yet, though payday was {date}.",
                 "I can't find my payslip for last month. Can HR re-issue it?",
                 "There's an unexplained deduction on my payslip. Please explain what it is.",
                 "My bank account changed. How do I update it for payroll before the next cycle?",
                 "The overtime I worked in {month} wasn't included in my pay.".replace("{month}", "February")],
        "detail": ["My employee ID is E{n}27.",
                   "The amount is short by about {amount}.",
                   "I've checked with my bank, nothing received.",
                   "Please look into it before the weekend."],
    },
    {   # benefits
        "subjects": ["Health insurance enrollment", "Add dependent to medical plan", "Provident fund contribution change",
                     "Dental coverage question", "Benefits enrollment deadline"],
        "core": ["I'd like to add my spouse to my company health insurance. What forms do I need?",
                 "When does the open enrollment window close for this year's benefits?",
                 "I want to change my provident fund contribution percentage starting next month.",
                 "Does our medical plan cover dental and vision, or is that a separate enrollment?",
                 "I got married recently and need to update my beneficiary details."],
        "detail": ["I joined in January.",
                   "Please send the enrollment form.",
                   "My employee ID is E{n}09."],
    },
    {   # employment verification
        "subjects": ["Employment verification letter", "Experience certificate request", "Salary certificate for loan",
                     "Letter for visa application", "Relieving letter"],
        "core": ["I need an employment verification letter for my home loan application.",
                 "Could HR issue my experience certificate? I left the company in {month}.".replace("{month}", "June"),
                 "My visa appointment needs a letter confirming my job title and salary. How soon can you provide it?",
                 "I'm still waiting for my relieving letter and the new employer needs it."],
        "detail": ["It should be on company letterhead.",
                   "I was employed from 2019 to 2023.",
                   "A scanned copy is fine for now."],
    },
    {   # resignation
        "subjects": ["Resignation notice", "Notice period question", "Full and final settlement",
                     "Last working day confirmation", "Exit formalities"],
        "core": ["I'm submitting my resignation. Please tell me the process and my last working day.",
                 "Can my notice period be shortened if my new employer needs me earlier?",
                 "When will my full and final settlement be processed after I leave?",
                 "What exit formalities do I need to complete before my last day?",
                 "Could you confirm how my unused leave will be paid out when I resign?"],
        "detail": ["My notice period is {n_days} days according to my contract.",
                   "I've already informed my manager.",
                   "I'd like a smooth handover."],
    },
    {   # workplace complaint
        "subjects": ["Complaint about a colleague", "Workplace harassment concern", "Manager conduct issue",
                     "Confidential grievance", "Unfair treatment at work"],
        "core": ["I want to report inappropriate behaviour from a coworker. What is the confidential process?",
                 "My manager keeps assigning me an unreasonable workload and ignores my concerns.",
                 "I've been excluded from team meetings and I think it's unfair. Can I raise a formal grievance?",
                 "I'd like to speak with someone from HR about a sensitive issue involving my team lead.",
                 "There was an incident in the office last week that made me uncomfortable and I want to document it."],
        "detail": ["I'd prefer to keep this private.",
                   "I have emails as evidence.",
                   "Please let me know who I can meet."],
    },
    {   # onboarding
        "subjects": ["Onboarding documents for new hire", "First day instructions", "Joining formalities",
                     "Documents to bring on joining day", "New joiner questions"],
        "core": ["I'm joining on {date} as {role}. Which documents should I bring on my first day?",
                 "Could you share the joining instructions and dress code for new employees?",
                 "When will I receive my offer letter countersignature and ID card?",
                 "I'm a new hire and haven't received my laptop or the welcome pack yet.",
                 "I have a few questions about the background verification form I was sent."],
        "detail": ["My offer letter is dated {date}.",
                   "I'm relocating from {city}.",
                   "Thank you, really excited to join."],
    },
    {   # policies, performance, training
        "subjects": ["Remote work policy", "Appraisal cycle dates", "Training budget reimbursement",
                     "Promotion eligibility", "Work from home days"],
        "core": ["Could you share the current remote work policy? I'd like to work from home two days a week.",
                 "When does the annual appraisal cycle begin and who conducts the review?",
                 "I completed a certification course. How do I claim reimbursement from the learning budget?",
                 "What are the criteria for promotion to senior level and how long does it usually take?",
                 "Is there an official policy on working from another city for a month?"],
        "detail": ["I've been in my current role for two years.",
                   "My manager approved the course.",
                   "A link to the handbook would be great."],
    },
    {   # referrals / internships
        "subjects": ["Internship opportunities", "Employee referral bonus", "Do you hire freshers?",
                     "Referring a friend for {role}", "Campus hiring inquiry"],
        "core": ["Are there any internships open for the summer? I'm a final-year computer science student.",
                 "I referred a friend for the {role} opening. How does the referral bonus work?",
                 "Do you hire fresh graduates and what is the usual selection process?",
                 "Our college placement cell would like to invite your recruiters for campus hiring."],
        "detail": ["I have attached my CV.",
                   "I graduate in May.",
                   "My friend's name is {name}."],
    },
    {   # tax forms
        "subjects": ["Tax forms for the year", "Tax declaration deadline", "Need Form 16", "W-2 not received"],
        "core": ["Where can I download my annual tax statement? I can't find it on the employee portal.",
                 "What is the last date to submit my investment declaration for tax savings?",
                 "I haven't received my Form 16 yet and the filing deadline is close.",
                 "My W-2 shows an incorrect address. Can HR correct and reissue it?"],
        "detail": ["My employee ID is E{n}33.",
                   "I've checked my spam folder."],
    },
]

GENERAL = [
    {   # product information
        "subjects": ["Question about features", "What does the {plan} plan include?", "Does your product support multiple languages?",
                     "Product compatibility question", "Quick question before I sign up"],
        "core": ["Which features come with the {plan} plan compared to Starter?",
                 "Does your product support right-to-left languages and Bengali?",
                 "Can your software be used offline, or does it need an internet connection all the time?",
                 "Before I sign up, can you tell me how many users can share one workspace?",
                 "Is there a free trial available and does it need a credit card?"],
        "detail": ["We are a small team of {n}.",
                   "I couldn't find this on your website.",
                   "Any comparison chart would help."],
    },
    {   # demo / sales
        "subjects": ["Request a demo", "Enterprise quote", "Volume licensing for {n} users", "Can we talk to sales?",
                     "Interested in a pilot"],
        "core": ["We're evaluating tools for a team of {n} people and would like to schedule a demo.",
                 "Do you offer discounts for educational institutions or NGOs?",
                 "Could someone from sales get in touch about an enterprise agreement for our group?",
                 "We'd like to run a two-week pilot with a few departments. How do we start?"],
        "detail": ["We're based in {city}.",
                   "Next week works best for a call.",
                   "My phone number is below."],
    },
    {   # contact, hours, locations
        "subjects": ["Support hours", "How can I reach a phone agent?", "Where is your office?",
                     "Contact details needed", "Holiday opening times"],
        "core": ["What are your customer support hours and are you open on weekends?",
                 "Is there a phone number I can call? I couldn't find one on the contact page.",
                 "Where is your head office located? I'd like to visit next month.",
                 "Are you closed over the holidays and when will the team be back?",
                 "Which email address should I use for general questions?"],
        "detail": ["I'm in a different time zone.",
                   "Thanks for the info.",
                   "Please reply by email."],
    },
    {   # feature requests
        "subjects": ["Feature suggestion: dark mode", "Would love bulk editing", "Suggestion for improvement",
                     "Idea for the next release", "Feature request"],
        "core": ["It would be really helpful if we could tag items in bulk instead of one at a time.",
                 "Please consider adding a calendar view. Many of us would use it daily.",
                 "Any plans to support exporting to Google Sheets directly?",
                 "I'd love a keyboard shortcut cheat sheet. It would make my workflow faster.",
                 "Suggestion: allow custom colours for labels so teams can match their own conventions."],
        "detail": ["Happy to share more on how we use it.",
                   "Is there a public roadmap?",
                   "Keep up the good work."],
    },
    {   # feedback / compliments / complaints
        "subjects": ["Great support experience", "Thank you", "Disappointed with response time",
                     "Feedback on your service", "Compliment for your team"],
        "core": ["I just wanted to say thank you. The agent I spoke with yesterday was helpful and patient.",
                 "I've been waiting several days for a reply to my earlier message. Slow responses are disappointing.",
                 "Your new interface looks fantastic, and the team deserves credit for it.",
                 "I'm sharing feedback after using the product for six months. Overall very happy, with a few small notes.",
                 "The onboarding call was confusing and I left with more questions than answers."],
        "detail": ["Please pass this on to the team.",
                   "Would be glad to take part in a survey.",
                   "I'd still recommend you to friends."],
    },
    {   # partnerships / press / vendors
        "subjects": ["Partnership inquiry", "Press inquiry", "Reseller program", "Sponsorship request",
                     "Collaboration proposal"],
        "core": ["We'd like to explore a partnership between our companies. Who is the right contact?",
                 "I'm a journalist writing about workplace software and would like to speak with a spokesperson.",
                 "Do you have a reseller or affiliate program for agencies?",
                 "We're organising a conference in {city} and would like to invite you as a sponsor.",
                 "Our firm provides integration services and wants to become an implementation partner."],
        "detail": ["Deadline is the end of the month.",
                   "Our website is below.",
                   "Happy to jump on a short call."],
    },
    {   # docs / how-to
        "subjects": ["How do I get started?", "Where can I find the user guide?", "Are there training webinars?",
                     "Looking for documentation", "Tutorial videos?"],
        "core": ["I'm new here. Is there a quick start guide or tutorial video?",
                 "Where can I find documentation about data retention and archiving?",
                 "Do you run onboarding webinars for new teams?",
                 "How do I invite a teammate to my workspace? I'm not sure where that option is.",
                 "Is there a community forum where users share tips?"],
        "detail": ["I looked at the help centre but wasn't sure.",
                   "We just signed up this week.",
                   "A link would be perfect."],
    },
    {   # privacy / legal / compliance
        "subjects": ["GDPR question", "Data processing agreement", "Do you have SOC 2?", "Where is data stored?",
                     "Privacy policy question"],
        "core": ["Could you send us your data processing agreement? Our legal team needs it for review.",
                 "Are you SOC 2 or ISO 27001 certified? Our security questionnaire asks for it.",
                 "In which country is customer data stored and can we choose the region?",
                 "How do I request a copy of the personal data you hold about me?",
                 "Does your privacy policy allow you to share usage data with third parties?"],
        "detail": ["We operate in the EU.",
                   "A short summary would help our compliance review.",
                   "Please point me to the right page."],
    },
    {   # events / newsletters
        "subjects": ["Unsubscribe from newsletter", "Webinar registration", "Product roadmap question",
                     "Event invitation", "Too many marketing emails"],
        "core": ["Please remove me from your newsletter list. I keep getting emails every week.",
                 "I signed up for the upcoming webinar but didn't get a confirmation email.",
                 "Where can I see what features are planned for next quarter?",
                 "Will your annual user conference be held online this year?",
                 "I'd like to get product announcements only, not promotions. Can I change my email preferences?"],
        "detail": ["Registered with my work email.",
                   "Thanks in advance.",
                   "Not urgent."],
    },
    {   # misc / verification of messages
        "subjects": ["Is this email really from you?", "Wrong number?", "Verify a message from your company",
                     "Who sent me this?", "Strange message from your brand"],
        "core": ["I received an email that claims to be from your company. How can I check if it's genuine?",
                 "Someone from your team called me earlier. Can you confirm if that was you?",
                 "I got a text with your brand name and a link. I don't want to click it before checking.",
                 "I think your reply was sent to me by mistake. It looks like it was meant for someone else."],
        "detail": ["I haven't opened any links.",
                   "I can forward the original message.",
                   "Thank you for confirming."],
    },
    {   # availability / regions / languages
        "subjects": ["Availability in India", "Do you support Bengali?", "Is the product available in my country?",
                     "Regional pricing and languages"],
        "core": ["Is your service available in India, and do you have local support in my time zone?",
                 "Does the interface support Bengali, Hindi or other regional languages?",
                 "Can customers in {city} use the product without a VPN?",
                 "Do you have a local office or representative in my region?"],
        "detail": ["We have team members in several countries.",
                   "Just checking before we commit."],
    },
]

SCENARIOS = {"Billing": BILLING, "Technical": TECHNICAL, "HR": HR, "General": GENERAL}

# Short side comments about another department, added to a small share of tickets.
CROSS_TOPIC = {
    "Billing": ["Also, the billing page was loading slowly when I tried.", "I also had trouble logging in earlier today.",
                "By the way, who handles questions about plans for bigger teams?"],
    "Technical": ["I'd also like a copy of my latest invoice while you're at it.", "Separately, do you offer a plan for larger teams?",
                  "Also my subscription was renewed today, can someone confirm the amount?"],
    "HR": ["The employee portal was slow too when I tried to check this.", "Also, my payment from the reimbursement system hasn't arrived.",
           "I also want to know if there is a general contact email for questions."],
    "General": ["I also noticed a charge on my card and I'm not sure what it is.", "Also, the login page was a bit slow earlier.",
                "My friend is also interested in a position there."],
}

VAGUE = [
    ("Question", "Hi, I have a question about my account. Could someone call me back when possible?"),
    ("Need help", "Hello, I need some help with something. Please get in touch."),
    ("Issue", "There's an issue that I would like to discuss with someone. Please let me know when you are available."),
    ("Follow up", "Following up on my previous message. Still waiting for an answer."),
    ("Help", "Can somebody assist me please?"),
    ("Request", "I sent a request earlier and wanted to check that it was received."),
    ("Urgent matter", "Please contact me about my case as soon as you can."),
    ("Hello", "Hi, who should I speak to about this? Thanks."),
    ("Query", "I would like more information, please reply to this email."),
    ("Problem", "Something went wrong and I'm not sure who to ask. Can you point me in the right direction?"),
    ("Need assistance", "I need assistance with my request, thank you for your time."),
    ("Re: my case", "Thanks for the update. Please keep me posted on the next steps."),
]

GREETINGS = ["Hi team,", "Hello,", "Hi,", "Good morning,", "Good afternoon,", "To whom it may concern,",
             "Dear Support,", "Hey,", "Hello there,", "Dear team,"]
CLOSINGS = ["Please advise.", "Thanks in advance.", "Let me know what you need from my side.",
            "I'd appreciate your help.", "Can someone look into this today?", "Thanks!", "Looking forward to your reply.",
            "Please get back to me as soon as possible.", "Thank you for your time.", "Please fix this immediately.",
            "This is urgent for us.", "It's quite critical for our work.", "Any help is appreciated."]
SIGNOFFS = ["Regards, {name}", "Thanks, {name}", "Best, {name}", "- {name}", "Sincerely, {name}", "Cheers, {name}"]
PREFIXES = ["Re: ", "Fwd: ", "URGENT: ", "Quick question - ", "[Support] ", "Hi - "]


# --------------------------------------------------------------------------
# Text assembly
# --------------------------------------------------------------------------
def add_typos(text: str, rng: random.Random) -> str:
    """Swap two neighbouring letters inside one or two longer words."""
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) > 5 and w.isalpha()]
    for i in rng.sample(candidates, k=min(len(candidates), rng.randint(1, 2))):
        w = words[i]
        j = rng.randint(1, len(w) - 3)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def build_ticket(department: str, rng: random.Random) -> tuple[str, str]:
    """Create one (subject, body) pair for the given department."""
    slots = make_slots(rng)

    # A small share of tickets are intentionally vague.
    if rng.random() < VAGUE_SHARE:
        subject, body = rng.choice(VAGUE)
        if rng.random() < 0.6:
            body = f"{rng.choice(GREETINGS)} {body}"
        if rng.random() < 0.4:
            body += f" {rng.choice(SIGNOFFS)}".format_map(slots)
        return subject, body

    scenario = rng.choice(SCENARIOS[department])
    subject = rng.choice(scenario["subjects"]).format_map(slots)
    core = rng.choice(scenario["core"]).format_map(slots)

    style = rng.random()
    if style < 0.08:  # terse: just the core sentence
        parts = [core]
    else:
        parts = []
        if rng.random() < 0.55:
            parts.append(rng.choice(GREETINGS))
        parts.append(core)
        details = scenario.get("detail", [])
        for d in rng.sample(details, k=rng.choice([0, 1, 1, 2]) if details else 0):
            parts.append(d.format_map(slots))
        if rng.random() < CROSS_TOPIC_SHARE:
            other = rng.choice([d for d in CROSS_TOPIC if d != department])
            parts.append(rng.choice(CROSS_TOPIC[other]))
        if rng.random() < 0.5:
            parts.append(rng.choice(CLOSINGS))
        if rng.random() < 0.3:
            parts.append(rng.choice(SIGNOFFS).format_map(slots))
    body = " ".join(parts)

    # Writing-style variations
    if rng.random() < 0.10:
        body = body.lower()
    if rng.random() < 0.08:
        body = add_typos(body, rng)
    if rng.random() < 0.06:
        subject = subject.upper()
    elif rng.random() < 0.10:
        subject = rng.choice(PREFIXES) + subject
    if rng.random() < 0.05:
        body = body.replace(".", "!", 1)
    return subject, body


def generate_dataset(seed: int = SEED) -> pd.DataFrame:
    """Build the full dataset and return it as a DataFrame."""
    rng = random.Random(seed)
    rows = []
    seen = set()
    for department, target in DEPARTMENT_TARGETS.items():
        made, attempts = 0, 0
        while made < target and attempts < target * 50:
            attempts += 1
            subject, body = build_ticket(department, rng)
            key = (subject.lower(), body.lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append({"subject": subject, "body": body, "department": department})
            made += 1
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "ticket_id", [f"TKT-{i:05d}" for i in range(1, len(df) + 1)])
    return df


def save_dataset(path: Path = DATA_PATH, seed: int = SEED) -> pd.DataFrame:
    """Generate the dataset and write it to CSV."""
    df = generate_dataset(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return df


if __name__ == "__main__":
    frame = save_dataset()
    print(f"Saved {len(frame)} tickets to {DATA_PATH}")
    print(frame["department"].value_counts().to_string())
