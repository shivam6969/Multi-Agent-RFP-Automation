import { Save } from "lucide-react";
import { useEffect, useState, type FormEvent } from "react";

import { useCompanyProfile, useSaveCompanyProfile } from "../api/hooks";
import type { CompanyProfile } from "../api/types";
import { Button, ErrorBox, Notice, PageHeader, Spinner } from "../components/ui";

type Field = { key: keyof CompanyProfile; label: string; wide?: boolean; textarea?: boolean; placeholder?: string };

const FIELDS: Field[] = [
  { key: "name", label: "Company name" },
  { key: "tagline", label: "Tagline", placeholder: "Shown under your name on the proposal" },
  { key: "about", label: "About the company", wide: true, textarea: true, placeholder: "Two or three sentences used in the proposal" },
  { key: "brands", label: "Brands you supply", placeholder: "e.g. Havells" },
  { key: "years_in_business", label: "Years in business" },
  { key: "address", label: "Address", wide: true },
  { key: "gstin", label: "GSTIN" },
  { key: "website", label: "Website" },
  { key: "email", label: "Email" },
  { key: "phone", label: "Phone" },
  { key: "signatory_name", label: "Signatory name" },
  { key: "signatory_title", label: "Signatory title" },
];

export default function Settings() {
  const { data, isLoading } = useCompanyProfile();
  const save = useSaveCompanyProfile();
  const [form, setForm] = useState<CompanyProfile | null>(null);

  useEffect(() => {
    if (data && !form) setForm(data);
  }, [data, form]);

  if (isLoading || !form) return <div className="flex justify-center p-16"><Spinner /></div>;

  function submit(e: FormEvent) {
    e.preventDefault();
    if (!form) return;
    const { is_demo: _demo, ...profile } = form;
    save.mutate({ ...profile, years_in_business: profile.years_in_business ? Number(profile.years_in_business) : null });
  }

  return (
    <>
      <PageHeader title="Settings" subtitle="Your company details, used as “Prepared by” on every proposal." />
      {data?.is_demo && (
        <div className="mb-5">
          <Notice tone="blue">These are placeholder details for a fictitious demo company. Replace them with your own before sending proposals.</Notice>
        </div>
      )}
      <form onSubmit={submit} className="card p-6">
        <div className="grid gap-4 sm:grid-cols-2">
          {FIELDS.map((f) => (
            <div key={f.key} className={f.wide ? "sm:col-span-2" : ""}>
              <label className="label" htmlFor={f.key}>{f.label}</label>
              {f.textarea ? (
                <textarea
                  id={f.key}
                  className="input min-h-24"
                  placeholder={f.placeholder}
                  value={String(form[f.key] ?? "")}
                  onChange={(e) => setForm({ ...form, [f.key]: e.target.value })}
                />
              ) : (
                <input
                  id={f.key}
                  className="input"
                  placeholder={f.placeholder}
                  inputMode={f.key === "years_in_business" ? "numeric" : undefined}
                  required={f.key === "name"}
                  value={String(form[f.key] ?? "")}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      [f.key]: f.key === "years_in_business" ? e.target.value.replace(/[^\d]/g, "") : e.target.value,
                    })
                  }
                />
              )}
            </div>
          ))}
        </div>
        {save.error && <div className="mt-4"><ErrorBox title="Could not save">{(save.error as Error).message}</ErrorBox></div>}
        <div className="mt-6 flex items-center justify-end gap-3">
          {save.isSuccess && <span className="text-sm text-emerald-600">Saved. New proposals will use these details.</span>}
          <Button type="submit" icon={<Save className="h-4 w-4" />} loading={save.isPending}>Save</Button>
        </div>
      </form>
    </>
  );
}
