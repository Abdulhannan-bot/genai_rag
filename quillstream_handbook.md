# Quillstream Customer Handbook

Version 4.2, effective 1 March 2026. This handbook describes plans, billing, limits, security, and support for Quillstream, a managed event-streaming platform. It applies to all customers unless a signed Enterprise contract states otherwise.

## 1. Platform Overview

### 1.1 What Quillstream is

Quillstream is a managed, Kafka-compatible event-streaming service. Customers publish messages to topics, and other applications read those messages through consumer groups. Quillstream operates the brokers, storage, and replication, so customers do not manage servers. The service is compatible with Kafka protocol 3.6 clients, which means most existing Kafka client libraries work without code changes.

### 1.2 Core concepts

A **topic** is a named stream of messages. Each topic is split into **partitions**, which allow messages to be read and written in parallel. Messages inside a single partition are strictly ordered, but ordering is not guaranteed across partitions. A **consumer group** is a set of applications that share the work of reading a topic; each partition is read by exactly one member of the group at a time. **Retention** is how long messages are kept before the oldest segments are deleted. Every topic is stored with a replication factor of 3, meaning three copies of each message exist in different availability zones.

## 2. Plans and Pricing

### 2.1 Plan comparison

Quillstream offers four plans. Prices are in US dollars and billed monthly unless the annual option is chosen.

| Plan | Monthly price | Included data transfer | Overage per GB | Max topics | Max partitions per topic | Max retention |
|---|---|---|---|---|---|---|
| Starter | $49 | 500 GB | $0.12 | 20 | 12 | 3 days |
| Growth | $399 | 5 TB | $0.09 | 200 | 48 | 14 days |
| Scale | $2,400 | 40 TB | $0.06 | 2,000 | 256 | 30 days |
| Enterprise | Custom | Custom | Custom | Unlimited | 1,024 | 365 days |

Overage is calculated on data transfer above the included monthly amount and is added to the next invoice.

### 2.2 Discounts

Customers on Starter, Growth, and Scale who pay annually receive two months free, which means they pay for 10 months and use 12. Quillstream's Startup Program gives eligible companies 50% off the Growth plan for the first 12 months, capped at $5,000 in total credits. The Education discount is 40% off Starter and Growth for accredited institutions. Discounts cannot be combined; if a customer qualifies for more than one, the largest single discount applies.

### 2.3 Changing plans

Upgrades take effect immediately and are prorated for the remainder of the billing cycle. Downgrades take effect at the start of the next billing cycle. A downgrade is blocked if the organization's current usage exceeds the limits of the target plan; for example, an organization with 35 topics cannot downgrade from Growth to Starter until it deletes at least 15 topics.

### 2.4 Free trial

New organizations can start a 14-day trial of the Growth plan without entering a credit card. The trial is limited to 10 GB of data transfer. A trial is not a paid subscription, so the refund policy in Section 3.2 does not apply to it.

## 3. Billing and Refunds

### 3.1 Invoicing and payment

Starter and Growth customers are charged by credit card on the billing date, which is the day the subscription started. Scale and Enterprise customers are invoiced on the 1st of each month with net 30 payment terms. If a card payment fails, Quillstream retries three times, after 1 day, 3 days, and 7 days. If the third retry fails, the account is suspended. Data in a suspended account is retained for 30 days and then permanently deleted. Paying the outstanding balance within that window restores access.

### 3.2 Refund policy (current)

The following policy has applied since 1 March 2026.

**Monthly plans (Starter and Growth).** A customer may request a full refund within 14 days of the first charge on a new subscription, provided that less than 100 GB of data has been transferred. After 14 days, or after 100 GB of transfer, the first charge is not refundable. Renewal charges are never refundable.

**Annual plans (Starter and Growth).** A customer may request a full refund within 30 days of purchase. After 30 days there are no refunds and no prorated refunds, but the customer can cancel to prevent the next renewal.

**Scale and Enterprise.** Refunds are governed by the signed contract. There is no self-serve refund for these plans.

**Overage charges** are non-refundable on every plan. Approved refunds are returned to the original payment method within 5 to 10 business days. Requests are submitted through the billing portal or by emailing billing@quillstream.example.

### 3.3 Refund policy (superseded)

Before 1 March 2026, monthly plans had a 30-day refund window with no data-transfer condition. Customers who subscribed before 1 March 2026 keep these older terms until their next renewal date, after which the current policy in Section 3.2 applies. This section is kept for reference only.

## 4. Service Level Agreement

### 4.1 Uptime commitments

Each plan has a monthly uptime commitment, measured across all production clusters in the organization.

| Plan | Monthly uptime commitment |
|---|---|
| Starter | None (best effort) |
| Growth | 99.9% |
| Scale | 99.95% |
| Enterprise | 99.99% (contractual) |

For reference, 99.9% allows about 43 minutes of downtime in a 30-day month, 99.95% allows about 22 minutes, and 99.99% allows about 4 minutes.

### 4.2 Service credits

If measured monthly uptime falls below the commitment, the customer is eligible for a service credit calculated as a percentage of that month's fee.

- Growth: 10% if uptime is below 99.9% but at least 99.0%; 25% if below 99.0%.
- Scale: 10% if uptime is below 99.95% but at least 99.9%; 25% if below 99.9% but at least 99.0%; 50% if below 99.0%.
- Enterprise: as defined in the signed contract.

Credits must be claimed within 30 days after the end of the affected month, are applied to future invoices, and are never paid out as cash. The maximum credit in any month is 50% of that month's fee. Scheduled maintenance does not count as downtime if it was announced at least 72 hours in advance and does not exceed 4 hours in a calendar month. Downtime caused by customer misconfiguration or force majeure events is also excluded.

## 5. API and Limits

### 5.1 Rate limits

| Plan | Produce requests per second | Fetch requests per second | Admin API calls per minute |
|---|---|---|---|
| Starter | 100 | 100 | 60 |
| Growth | 1,000 | 1,000 | 300 |
| Scale | 10,000 | 10,000 | 1,200 |
| Enterprise | Custom | Custom | Custom |

Requests above the limit are rejected with HTTP status 429 and error code ERR_4291. The response includes a Retry-After header stating how many seconds to wait before trying again.

### 5.2 Error codes

| Code | HTTP | Meaning | Recommended action |
|---|---|---|---|
| ERR_4291 | 429 | Rate limit exceeded | Wait for the Retry-After interval |
| ERR_4031 | 403 | API key lacks the required scope | Create a key with the needed scope |
| ERR_4091 | 409 | Topic already exists | Use a different topic name |
| ERR_4093 | 409 | Topic quota exceeded for the plan | Delete unused topics or upgrade |
| ERR_4132 | 413 | Message exceeds the maximum size | Reduce the message size |
| ERR_4221 | 422 | Requested retention exceeds the plan maximum | Lower the retention setting |
| ERR_5003 | 503 | Partition leader temporarily unavailable | Retry with exponential backoff |

### 5.3 Message size

The maximum message size is 1 MB on Starter and Growth, 4 MB on Scale, and 8 MB on Enterprise. Larger payloads should be stored externally, with only a reference sent through Quillstream.

### 5.4 Retry guidance

For ERR_5003 and ERR_4291, clients should retry using exponential backoff with jitter. The recommended settings are an initial delay of 200 milliseconds, a multiplier of 2, a maximum of 5 attempts, and random jitter of up to 20% of each delay. Do not retry ERR_4031, ERR_4091, ERR_4093, ERR_4132, or ERR_4221, because repeating the same request will fail again.

```python
import random
import time

def send_with_retry(client, message, max_attempts=5, base_delay=0.2):
    for attempt in range(max_attempts):
        try:
            return client.produce("orders", message)
        except TransientError:
            delay = base_delay * (2 ** attempt)
            time.sleep(delay + random.uniform(0, delay * 0.2))
    raise RuntimeError("Gave up after 5 attempts")
```

### 5.5 Authentication

API keys start with qs_live_ for production clusters and qs_test_ for sandbox clusters. A key is displayed only once at creation, so it must be copied immediately. Each key has one or more scopes: produce, consume, or admin. Quillstream recommends rotating keys every 90 days. Enterprise organizations can enforce a mandatory rotation period.

## 6. Security and Access Control

### 6.1 Roles and permissions

| Role | Manage topics | Manage members | View billing | Delete the organization |
|---|---|---|---|---|
| Owner | Yes | Yes | Yes | Yes |
| Admin | Yes | Yes | No | No |
| Developer | Yes | No | No | No |
| Billing | No | No | Yes | No |
| Viewer | No | No | No | No |

Viewers have read-only access to metrics dashboards. The Billing role can view invoices and update payment methods but cannot see topic data. Every organization must have at least one Owner.

### 6.2 Encryption

All traffic uses TLS 1.3, and TLS 1.2 is accepted as a minimum for older clients. Data at rest is encrypted with AES-256. Customer-managed encryption keys (BYOK) are available only on the Enterprise plan.

### 6.3 Compliance and data residency

Quillstream holds SOC 2 Type II and ISO 27001 certifications. The most recent SOC 2 audit period ended in December 2025, and audits are repeated annually. A GDPR data processing agreement is available to every customer on request. A HIPAA business associate agreement is available on the Enterprise plan only. Clusters can be created in three regions: us-east-1 (Virginia), eu-west-1 (Ireland), and ap-south-1 (Mumbai). The region is fixed at cluster creation and cannot be changed afterwards; moving to another region requires creating a new cluster and mirroring the topics.

## 7. Data Retention and Backups

### 7.1 Retention and deletion

Each topic has a configurable retention period up to the maximum for the plan. When messages exceed the retention period, the oldest segments are deleted automatically. A deleted topic enters a soft-delete state and can be recovered for 7 days on Growth, Scale, and Enterprise, and for 48 hours on Starter. After the soft-delete window the topic and its messages are permanently removed. When an organization is terminated, all of its data is deleted within 30 days. Scale and Enterprise customers can request a certificate of deletion.

### 7.2 Backups

Starter and Growth topics are protected only by replication (three copies) and are not backed up separately. Scale and Enterprise clusters also receive daily snapshots, which are kept for 7 days. Replication protects against hardware failure but not against accidental topic deletion or incorrect writes, so customers with strict recovery needs should use Scale or higher.

## 8. Support and Incident Response

### 8.1 Severity levels and response targets

A P1 incident means production is down or data is at risk. A P2 incident means production is degraded but working. A P3 request is a general question or a minor issue.

| Plan | Channel | P1 response | P2 response | P3 response |
|---|---|---|---|---|
| Starter | Community forum | No guaranteed response | No guaranteed response | No guaranteed response |
| Growth | Email | 4 business hours | 1 business day | 2 business days |
| Scale | 24x7 chat and email | 1 hour | 4 hours | 1 business day |
| Enterprise | 24x7 phone and chat, plus a dedicated Technical Account Manager | 15 minutes | 1 hour | 4 hours |

### 8.2 Escalation process

1. Check the status page at status.quillstream.example for a known incident.
2. Open a ticket in the support portal with the cluster ID, region, time of first occurrence, and any error codes.
3. Choose the severity that matches the business impact. Tickets that are over-classified may be downgraded by support.
4. If the response target is missed, Scale and Enterprise customers can click Escalate in the ticket, which pages the on-call engineering manager.
5. Enterprise customers can also contact their Technical Account Manager directly.

Quillstream publishes a post-incident report within 5 business days for any P1 incident that affected more than one customer.

## 9. Getting Started

Follow these steps to send your first message.

1. Create an organization and choose a region.
2. Create a cluster, which takes about 6 minutes to provision.
3. Create an API key with the scopes you need.
4. Install the command-line tool with `pip install qs-cli`.
5. Create your first topic.

```bash
qs-cli login --key qs_live_xxxxxxxx
qs-cli topic create orders --partitions 6 --retention 7d
```

Note: on the Starter plan, a 7-day retention setting is rejected with ERR_4221 because the Starter maximum is 3 days. Use `--retention 3d` or upgrade the plan.

## 10. Frequently Asked Questions

### Does Quillstream host Kafka Streams or Kafka Connect?

No. Quillstream supports Kafka protocol 3.6 clients, but Kafka Streams and Kafka Connect are not hosted by Quillstream. Customers run them in their own infrastructure.

### Can Quillstream be installed on-premises?

No. Quillstream is available only as a managed cloud service.

### Can I use one API key for both production and sandbox?

No. A qs_live_ key works only with production clusters and a qs_test_ key works only with sandbox clusters.

### Are overage charges refundable?

No. Overage charges are non-refundable on every plan.

## 11. Changelog

- Version 4.2 (1 March 2026): New refund policy in Section 3.2. Maximum message size on Scale raised from 2 MB to 4 MB.
- Version 4.1 (November 2025): Added the ap-south-1 (Mumbai) region.
- Version 4.0 (August 2025): Growth plan price increased from $349 to $399 per month. Starter maximum retention reduced from 7 days to 3 days.