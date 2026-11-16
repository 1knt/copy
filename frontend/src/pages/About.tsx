import { Panel } from '../components'
import { ASSIGNMENT_2_ONLY } from '../scope'

const TEAM = ['Dang Minh Bui', 'Phan Hien Hoang', 'Van Minh Nguyen']

const ALL_STAGES = [
  { phase: 'A1', title: 'Project management and prototype', detail: 'Plan, requirements, and this interface running against the real API contract.' },
  { phase: 'A2', title: 'Data preparation and modelling', detail: 'Two open datasets merged and deduplicated, then classification and clustering with a recorded evaluation.' },
  { phase: 'A3', title: 'Integrated web application', detail: 'The frozen models served through FastAPI and read by this React client.' },
]
const STAGES = ASSIGNMENT_2_ONLY ? ALL_STAGES.filter(stage => stage.phase !== 'A3') : ALL_STAGES

export default function About() {
  return (
    <div className="rail">
      <div className="nameplate">
        <span className="eyebrow micro">ABOUT THE PROJECT</span>
        <h1>Small messages.<br />Meaningful signals.</h1>
        <p className="lede">
          SMS spam screening for COS30049, group HN1_8.
        </p>
      </div>
      <div className="datum" />

      <div className="two-col">
        <Panel index="01" title="The project team">
          <div className="panel-body">
            <span className="micro">GROUP HN1_8</span>
            <ul className="roster">
              {TEAM.map((name, index) => (
                <li key={name}>
                  <span>{String(index + 1).padStart(2, '0')}</span>
                  {name}
                </li>
              ))}
            </ul>
          </div>
        </Panel>

        <Panel index="02" title="Where the work stands">
          <div className="panel-body">
            <ul className="roster">
              {STAGES.map(stage => (
                <li key={stage.phase} style={{ display: 'block' }}>
                  <span style={{ marginRight: 14 }}>{stage.phase}</span>
                  <b style={{ fontWeight: 550 }}>{stage.title}</b>
                  <p className="field-help" style={{ marginTop: 4 }}>{stage.detail}</p>
                </li>
              ))}
            </ul>
          </div>
        </Panel>
      </div>

      <div className="two-col" style={{ marginTop: 'clamp(16px, 2.4vw, 22px)' }}>
        {!ASSIGNMENT_2_ONLY && (
        <Panel index="03" title="How your message is handled">
          <div className="panel-body">
            <ul className="bullets">
              <li>A message is sent to the configured API only when you select Analyze, or automatically after you stop typing while Real-time is enabled.</li>
              <li>This interface stores no message text. Each session record keeps the spam score, the label, the model version, the decision threshold, the server request identifier, the character count and the time the reading was taken — and those same fields are what an export contains.</li>
              <li>Session records live in this tab's session storage. Clearing the session, or closing the tab, removes them. Your theme and real-time preferences are kept in local storage.</li>
              <li>Request identifiers exist for diagnostics. They do not imply that any message was written to a database.</li>
              <li>Message text is rendered as text, never as markup, and links are never opened for you.</li>
              <li>Fonts are served from this application, not a font CDN, so opening a page contacts no third party at all. Every network request the interface makes goes to its own API.</li>
            </ul>
          </div>
        </Panel>
        )}

        <Panel index={ASSIGNMENT_2_ONLY ? '03' : '04'} title="What this tool does not do">
          <div className="panel-body">
            <ul className="bullets">
              <li>It does not prove a message or link is safe.</li>
              <li>It does not scan files for malware.</li>
              <li>It covers English SMS only.</li>
            </ul>
          </div>
        </Panel>
      </div>
    </div>
  )
}
