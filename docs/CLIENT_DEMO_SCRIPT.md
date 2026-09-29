# Client Demo Script

These utterances demonstrate coverage only. The assistant uses OpenAI for natural
language understanding; none of these sentences is an application command rule.

## Employee

### Profile

**Example utterance only — not hardcoded**

English: “Show my employee profile.”

**Example utterance only — not hardcoded**

Arabic: “اعرض لي ملفي الوظيفي.”

### Vacation balance

**Example utterance only — not hardcoded**

English: “How much vacation balance do I have?”

**Example utterance only — not hardcoded**

Arabic: “وش باقي لي من رصيد الإجازة؟”

### Service duration

**Example utterance only — not hardcoded**

English: “How long have I been with the company?”

**Example utterance only — not hardcoded**

Arabic: “كم مدة خدمتي في الشركة؟”

### Attendance

**Example utterance only — not hardcoded**

English: “Show my attendance this month.”

**Example utterance only — not hardcoded**

Arabic: “أبغى أشوف حضوري هذا الشهر.”

### Missing punch

Use this section only when ResourcePlus has live missing-punch suggestions.

**Example utterance only — not hardcoded**

English: “Help me correct my missing punch from yesterday.”

**Example utterance only — not hardcoded**

Arabic: “ساعدني أصحح البصمة الناقصة أمس.”

### Leave or Business Travel

**Example utterance only — not hardcoded**

English: “Request Business Travel for 22 September 2026.”

**Example utterance only — not hardcoded**

Arabic: “قدّم لي طلب مهمة عمل ليوم 22 سبتمبر 2026.”

Review the generated confirmation before replying. Do not confirm against live
ResourcePlus unless the transaction is intentionally part of the demo.

### Request status

**Example utterance only — not hardcoded**

English: “Show my requests this month.”

**Example utterance only — not hardcoded**

Arabic: “وش حالة طلباتي لهذا الشهر؟”

Request status uses the complete calendar month; attendance stops at today.

### Cancellation

**Example utterance only — not hardcoded**

English: “Cancel my pending Business Travel request for 22 September 2026.”

**Example utterance only — not hardcoded**

Arabic: “ألغِ طلب مهمة العمل المعلّق ليوم 22 سبتمبر 2026.”

### Notifications

**Example utterance only — not hardcoded**

English: “Do I have unread HR notifications?”

**Example utterance only — not hardcoded**

Arabic: “هل عندي تنبيهات موارد بشرية ما قريتها؟”

**Example utterance only — not hardcoded**

English: “Mark my latest notification as read.”

**Example utterance only — not hardcoded**

Arabic: “علّم آخر تنبيه عندي كمقروء.”

Marking notifications read requires explicit confirmation.

## Manager

For manager demonstrations, sign in as the manager so the frontend sends that
manager's `email` + `instance` pair. Supervisor tools use this request-scoped
identity; they do not substitute an employee or configured manager identity.

### Pending approvals

**Example utterance only — not hardcoded**

English: “Show my pending team approvals.”

**Example utterance only — not hardcoded**

Arabic: “اعرض طلبات فريقي المعلّقة للموافقة.”

### Approve or reject one

**Example utterance only — not hardcoded**

English: “Approve Fahad’s Business Travel request.”

**Example utterance only — not hardcoded**

Arabic: “وافق على طلب مهمة العمل الخاص بفهد.”

### Bulk approval

Use only when safe test data exists and the displayed count and scope are correct.

**Example utterance only — not hardcoded**

English: “Approve all pending absence requests.”

**Example utterance only — not hardcoded**

Arabic: “وافق على جميع طلبات الغياب المعلّقة.”

## Voice

### English free-form

**Example utterance only — not hardcoded**

English: “Could you tell me whether I have any new HR alerts?”

### Saudi Arabic free-form

**Example utterance only — not hardcoded**

Arabic: “ممكن تقول لي إذا وصلني شيء جديد من الموارد البشرية؟”

### Natural confirmation

The confirmation classifier understands free-form English, Saudi conversational
Arabic, formal Arabic, and reasonable code-switching. The examples below are not
the only accepted replies.

**Example utterance only — not hardcoded**

English: “Yes, that looks right—go ahead.”

**Example utterance only — not hardcoded**

Arabic: “إيه تمام، توكل.”

**Example utterance only — not hardcoded**

English: “No, leave it unchanged.”

**Example utterance only — not hardcoded**

Arabic: “لا خلاص، لا تغيّر شيء.”
