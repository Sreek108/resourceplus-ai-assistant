import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import ChatMessage from "./ChatMessage";
import ResponseBlocks from "./ResponseBlocks";


describe("ResponseBlocks", () => {
  it("renders key-value data as a compact details grid", () => {
    const { container } = render(
      <ResponseBlocks blocks={[{
        type: "key_value",
        title: "Profile",
        items: [
          { label: "Employee code", value: "UN003" },
          { label: "Name", value: "Saneesh" },
          { label: "Company", value: "Universal" },
        ],
      }]} />,
    );

    expect(screen.getByRole("heading", { name: "Profile" })).toBeTruthy();
    expect(container.querySelector(".details-grid")).toBeTruthy();
    expect(container.querySelectorAll(".details-grid > div")).toHaveLength(3);
    expect(container.querySelector(".stat-card")).toBeNull();
  });

  it("renders stat cards as distinct KPI cards", () => {
    const { container } = render(
      <ResponseBlocks blocks={[{
        type: "stat_cards",
        title: "Allowance",
        items: [
          { label: "Remaining", value: 120 },
          { label: "Used", value: 0 },
          { label: "Limit", value: 120 },
        ],
      }]} />,
    );

    expect(container.querySelector(".stat-card-grid")).toBeTruthy();
    expect(container.querySelectorAll(".stat-card")).toHaveLength(3);
    expect(screen.getByText("Remaining")).toBeTruthy();
    expect(screen.getAllByText("120")).toHaveLength(2);
  });

  it("renders a scrollable semantic table with aligned trusted columns", () => {
    const { container } = render(
      <ResponseBlocks blocks={[{
        type: "table",
        title: "Attendance",
        columns: [
          { key: "date", label: "Date" },
          { key: "status", label: "Status" },
        ],
        rows: [{ date: "01/10/2026", status: "Regular" }],
      }]} />,
    );

    expect(screen.getByRole("region", { name: "Attendance table" }).tabIndex).toBe(0);
    expect(screen.getByRole("columnheader", { name: "Date" })).toBeTruthy();
    expect(screen.getByRole("cell", { name: "01/10/2026" })).toBeTruthy();
    expect(container.querySelector(".response-table-wrap")).toBeTruthy();
  });

  it("formats only the known correctable boolean as availability", () => {
    render(
      <ResponseBlocks blocks={[{
        type: "table",
        title: "Missing punches",
        columns: [
          { key: "correctable", label: "Correctable" },
          { key: "night_shift", label: "Night shift" },
        ],
        rows: [
          { correctable: true, night_shift: true },
          { correctable: false, night_shift: false },
        ],
      }]} />,
    );

    expect(screen.getByText("Available")).toBeTruthy();
    expect(screen.getByText("Not available")).toBeTruthy();
    expect(screen.getByRole("cell", { name: "true" })).toBeTruthy();
    expect(screen.getByRole("cell", { name: "false" })).toBeTruthy();
    expect(screen.queryByRole("cell", { name: /^trueAvailable$/ })).toBeNull();
  });

  it("renders friendly employee-request statuses as badges", () => {
    const { container } = render(
      <ResponseBlocks blocks={[{
        type: "table",
        title: "My requests",
        columns: [
          { key: "request", label: "Request" },
          { key: "date", label: "Date" },
          { key: "detail", label: "Detail" },
          { key: "status", label: "Status" },
        ],
        rows: [
          { request: "Leave", date: "2026-10-09", detail: "Compensatory Leave", status: "Approved" },
          { request: "Attendance correction", date: "2026-09-02", detail: "Outside Work", status: "Pending" },
        ],
      }]} />,
    );

    expect(screen.getByText("Approved").classList).toContain("status-approved");
    expect(screen.getByText("Pending").classList).toContain("status-pending");
    expect(container.querySelectorAll(".request-status-badge")).toHaveLength(2);
    expect(document.body.textContent).not.toContain("requestId");
  });

  it("keeps every long-table row and toggles its presentation", () => {
    const rows = Array.from({ length: 42 }, (_, index) => ({
      date: `Attendance day ${index + 1}`,
    }));
    const { container } = render(
      <ResponseBlocks blocks={[{
        type: "table",
        title: "Attendance",
        columns: [{ key: "date", label: "Date" }],
        rows,
      }]} />,
    );

    const table = screen.getByRole("table");
    expect(table.dataset.totalRows).toBe("42");
    expect(container.querySelectorAll("tbody tr")).toHaveLength(10);
    expect(screen.queryByText("Attendance day 42")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "View all (42)" }));
    expect(container.querySelectorAll("tbody tr")).toHaveLength(42);
    expect(screen.getByText("Attendance day 42")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Show less" }));
    expect(container.querySelectorAll("tbody tr")).toHaveLength(10);
    expect(table.dataset.totalRows).toBe("42");
  });

  it("keeps action labels visible and submits the unchanged action object", () => {
    const onAction = vi.fn();
    const action = {
      label: "22 Sep · Family Circumstances · Early Departure",
      value: "Select exceptional entry ce-private",
      style: "primary",
    };
    render(
      <ResponseBlocks
        blocks={[{ type: "actions", title: "Choose a request", actions: [action] }]}
        onAction={onAction}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: action.label }));
    expect(onAction).toHaveBeenCalledWith(action);
    expect(document.body.textContent).not.toContain("ce-private");
  });

  it("renders mobile-usable per-request actions without exposing trusted IDs", () => {
    const onAction = vi.fn();
    const approve = {
      label: "Approve",
      value: "Approve request 1",
      style: "primary",
      payload: { kind: "pending_approval", decision: "approve", ordinal: 1 },
    };
    const reject = {
      label: "Reject",
      value: "Reject request 1",
      style: "danger",
      payload: { kind: "pending_approval", decision: "reject", ordinal: 1 },
    };
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 390 });

    const { container } = render(
      <ResponseBlocks
        blocks={[{
          type: "table",
          title: "Pending approvals",
          columns: [
            { key: "employee", label: "Employee" },
            { key: "request", label: "Request" },
            { key: "date", label: "Date" },
            { key: "detail", label: "Detail" },
            { key: "status", label: "Status" },
            { key: "action", label: "Action" },
          ],
          rows: [{
            employee: "Talal Sabbagh",
            request: "Leave",
            date: "Oct 9",
            detail: "Compensatory Leave",
            status: "Pending",
            action: "",
          }],
          row_actions: [[approve, reject]],
        }]}
        onAction={onAction}
      />,
    );

    expect(container.querySelector(".table-row-actions")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Approve" }));
    expect(onAction).toHaveBeenCalledWith(approve);
    expect(screen.getByRole("button", { name: "Reject" })).toBeTruthy();
    expect(document.body.textContent).not.toContain("requestId");
    expect(document.body.textContent).not.toContain("rp-secret-id");
  });

  it("preserves RTL semantics on structured blocks", () => {
    const { container } = render(
      <ResponseBlocks
        direction="rtl"
        language="ar"
        blocks={[{
          type: "list",
          title: "الإشعارات",
          items: ["لديك إشعار جديد"],
        }]}
      />,
    );

    const blocks = container.querySelector(".response-blocks");
    expect(blocks.dir).toBe("rtl");
    expect(blocks.lang).toBe("ar");
  });

  it("localizes the known availability presentation in Arabic", () => {
    render(
      <ResponseBlocks
        direction="rtl"
        language="ar"
        blocks={[{
          type: "table",
          title: "البصمات المفقودة",
          columns: [{ key: "correctable", label: "قابل للتصحيح" }],
          rows: [{ correctable: true }, { correctable: false }],
        }]}
      />,
    );

    expect(screen.getByText("متاح")).toBeTruthy();
    expect(screen.getByText("غير متاح")).toBeTruthy();
    expect(screen.queryByText("Available")).toBeNull();
  });
});


describe("structured confirmation", () => {
  it("shows the trusted summary while keeping confirm and cancel callbacks", () => {
    const onConfirm = vi.fn();
    const onCancel = vi.fn();
    render(
      <ChatMessage
        message={{
          role: "assistant",
          text: "Please review your request.",
          requiresConfirmation: true,
          blocks: [{
            type: "confirmation",
            title: "Confirmation required",
            summary: "Submit Holiday for 20 October 2026?",
            actions: [
              { label: "Confirm", value: "confirm", style: "primary" },
              { label: "Cancel", value: "cancel", style: "secondary" },
            ],
          }],
        }}
        onConfirm={onConfirm}
        onCancel={onCancel}
      />,
    );

    expect(screen.getByText("Submit Holiday for 20 October 2026?")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Confirm" }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(onCancel).toHaveBeenCalledTimes(1);
  });
});
