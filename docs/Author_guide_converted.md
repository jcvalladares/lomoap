# Guide to EXAMPLE Author

> Converted and reorganized notes from `Author_full.txt` — a readable reference for EXAMPLE Author (ManuScript authoring tool).

## Summary
This document reorganizes the supplied `Author_full.txt` content into a readable Markdown reference for EXAMPLE Author (the "Author" application). It identifies main sections, subsections, UI flows, important commands/shortcuts, and notes where the original text references XML/HTML/JSON/JavaScript artifacts.

---

## Table of Contents
- Getting Started
- Best Practices
- ManuScripts
- ManuScript Catalog
  - Linking / Import / Export / Remove
  - Importing Elements
  - Catalog Keys & Filtering
- Fields View
  - Controls, Toolbar, Keyboard Shortcuts
  - Creating / Modifying / Deleting / Duplicating / Renaming
- Page Designer, Forms Mapper, Data Tester (overview)
- Printing & Reports
- Deploying a ManuScript
- Inheritance
- Appendix: File‑type Notes & Examples

---

## Getting Started
**Purpose**
- `Author` defines insurance product features (rates, rules, algorithms, docs, screen flow, policy transactions) using ManuScripts (XML product definitions).

**Primary Views**
- **Fields** — Define fields, groups, relationships.
- **Tables** — Define data tables.
- **Page Designer** — Create/edit input screens.
- **Data Tester** — Run and test ManuScripts using data documents.
- **Forms Mapper** — Create/edit forms and map them.

**Login Flow & Modes**
- Start `Author` from the Windows menu and use the Login dialog to choose an EXAMPLE Server or work locally (standalone).
- Use File → Login User/Server to change server or user while running.
- **Enterprise Mode**: When working against an EXAMPLE Server. Prompts for change confirmation; revision tracking may require comments.

**Directories & Options**
- Options → Directories configures where Catalog, Debug, Forms, Preview (XSL), Schema, etc. reside.
- Use **Restore Default Catalogs** to revert directory settings.

**Toolbar & UI**
- Toolbar contains common actions: New, Open, Save, Print, Validate, view buttons, and a Versions list.
- The Versions list shows all versions of the active ManuScript and inherited ManuScripts.

**Selected Keyboard Shortcuts**
```
F1   - Help Contents
F3   - Find next
F5   - Refresh
F7   - Search Viewer
F8   - Search Viewer / Constant Viewer
Ctrl+C - Copy
Ctrl+F - Find
Ctrl+N - New ManuScript
Ctrl+O - Open ManuScript
Ctrl+P - Print
Ctrl+S - Save
Ctrl+U - Used by
Ctrl+Z - Undo
Ctrl+Tab - Switch views
Shift+Enter - Insert line break
```

---

## Best Practices for ManuScript Development
- Use `data` as the root group name.
- Save frequently — there is no autosave.
- Add three private groups under public groups: *Input*, *Output*, and *Private* for organization.
- Use the `Prepend group name to field` option to avoid name collisions.
- Before changing a field, run `Used by` to locate dependencies.
- For large table content, prefer copy/paste from electronic lists instead of manual typing.
- Do not change `versionID` for versioned ManuScripts — it links versions together.

---

## ManuScripts
**Definition & Purpose**
- ManuScripts are XML files that contain fields, groups, tables, pages, forms, and other metadata describing a product.
- They define data paths, validations, UI rendering, and report generation.

**Structure**
- Groups (public/private), fields, pages, forms, and tables.
- Public items are included in output XML and downstream systems; private items remain internal.

---

## The ManuScript Catalog
**Role**
- Stores/manages ManuScripts, versions, metadata, and their locations.

**Common Actions**
- **Open / New**: via Open/New ManuScript dialog.
- **Link**: create a reference to an external ManuScript (shortcut-style).
- **Import**: copy a ManuScript (or elements) into the Catalog.
- **Export**: copy a ManuScript out of the Catalog to a chosen folder.
- **Remove**: remove a link or delete an imported file.

**Catalog Keys & Filtering**
- Default keys: `lob` (line of business) and `state`.
- Keys create the directory structure; you can add/remove keys and filter the Catalog view.

**Importing Elements**
- File → Import → Catalog → select ManuScript → choose which element type (Fields/Groups, Tables, Pages, Forms).
- The Import Wizard allows multi-select and will avoid duplicates unless forced.

---

## Fields View
**Description**
- Lists all groups and fields with structural navigation and manipulation tools.

**Toolbar & Controls**
- Buttons: Add Field, Add Group, Duplicate, Delete, Move, Find, Back, Refresh.
- Filters: Public, Private, Local Only, Worksheet.

**Field/Group Operations**
- **Add Field** (Ctrl+A) / **Add Group** (Ctrl+G).
- Field properties: Name, Visibility, Caption, Value Type, Data Type, Create-in Group.
- Group properties: Visibility, Class, Maximum, Required, Related Group.
- **Delete** prompts for confirmation.
- **Duplicate** copies group/field (groups duplicate nested items).
- **Rename** can adjust data path elements and optionally prepend group name.

**Data Mart Attributes (DB mapping / shredding)**
- Configure table mapping, primary keys, foreign keys, and parent relationships for public groups and fields used in shredded DB exports.

---

## Page Designer, Forms Mapper, Data Tester (Overview)
- Page Designer: WYSIWYG page layout and preview for pages described by ManuScripts.
- Forms Mapper: Map fields/pages to forms for rendering and publishing.
- Data Tester: Execute ManuScripts against sample data documents and validate behavior.

---

## Printing & Reports
- Standard reports: Fields List, Tables List, Calculations List, Public Groups and Fields List.
- Print flow: choose report → Preview → Print or Print-to-File.
- Custom reports: provide an XSL template for rendering; preview and print accordingly.

---

## Deploying a ManuScript
- Use Deployment Manager to deploy a ManuScript to one or more EXAMPLE servers.
- Optionally include dependencies.
- Deployment Manager shows the XML response from the server.

---

## Inheritance
**Concept**
- ManuScripts can inherit from base ManuScripts for Data, Pages, and Forms.
- Inheritance reduces duplication — derived ManuScripts extend base ones.

**Caveats**
- Overridden items in derived ManuScripts are not automatically updated when the base changes — manual reconciliation may be required.
- Use the Inheritance dialog or the Inheritance tab in Properties to inspect the chain.
- A ManuScript may inherit up to three other ManuScripts (one per category).

---

## Appendix: File‑type Notes & Examples
**References in the original text** (no explicit code blocks were present in the source):
- XML: ManuScripts are XML files; schemas are used for paths and validation.
- XSL: Preview and print use `.xsl` templates.
- TSV: Debug outputs and debug folder use `.tsv` files.
- HTML: Page previews and exported pages reference `.html`.
- PDF: Printed exports or Save Page to .PDF flows produce PDF files.

**Illustrative examples** (generated for clarity):

_Example Manuscript XML skeleton (illustrative):_
```xml
<manuscript id="12345" version="1.0">
  <properties>
    <caption>Example Product</caption>
    <culture>en-US</culture>
  </properties>
  <groups>
    <group name="data" visibility="public">
      <field name="PolicyNumber" dataType="string" visibility="public" />
    </group>
  </groups>
  <pages>
    <!-- Page definitions go here -->
  </pages>
</manuscript>
```

_Example XSL preview snippet (illustrative):_
```xml
<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:template match="/">
    <html><body>
      <h1><xsl:value-of select="manuscript/properties/caption"/></h1>
    </body></html>
  </xsl:template>
</xsl:stylesheet>
```

---

## Files & Paths Mentioned
- `pages_pdf` / `pages_pdf_txt`
- `pages_txt` / `pages_md`
- Catalog directories (catalog root path)
- `pages_md/merged/<docx_filename>`
- Debug folder: `.tsv` files
- Preview XSL files: `.xsl`

---

## Next Steps (optional actions I can take)
- Extract real ManuScript XML files from your repo and insert real code samples into this document.
- Save this file in a different path or commit it to a branch and open a PR.
- Convert additional `.chm` or attached docs into structured Markdown using the same template.

---

*Document generated from provided `Author_full.txt`. If you want changes to structure, more code examples, or automatic extraction of XML samples from repository files, tell me which files to include.*
