from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Type
import xml.etree.ElementTree as ET

from src.types.session import Session, SessionEvent
from src.types.events import EventBody, ALL_EVENT_TYPES

# Map tag name -> class, e.g. "HumanMsg" -> HumanMsg
_EVENT_TYPE_BY_NAME: Dict[str, Type[Any]] = {t.__name__: t for t in ALL_EVENT_TYPES}


def to_xml_string(session: Session) -> str:
    """
    Serialize Session -> XML string.

    Schema (human-editable):
      <session id="...">
        <event id="...">
          <HumanMsg>
            <content>...</content>
          </HumanMsg>
        </event>
        ...
      </session>
    """
    root = ET.Element("session", attrib={"id": session.session_id})

    for ev in session.events:
        ev_el = ET.SubElement(root, "event", attrib={"id": ev.event_id})
        _body_to_xml(parent=ev_el, body=ev.body)

    _indent(root)
    return ET.tostring(root, encoding="unicode")


def from_xml_string(xml_str: str) -> Session:
    """
    Deserialize XML string -> Session.
    """
    root = ET.fromstring(xml_str)
    if root.tag != "session":
        raise ValueError(f"Expected <session>, got <{root.tag}>")

    session_id = root.attrib.get("id")
    if not session_id:
        raise ValueError("Missing session id attribute: <session id='...'>")

    events: list[SessionEvent] = []
    for ev_el in root.findall("event"):
        event_id = ev_el.attrib.get("id")
        if not event_id:
            raise ValueError("Missing event id attribute: <event id='...'>")

        # Expect exactly one child under <event>: the body tag (e.g. <HumanMsg>...</HumanMsg>)
        body_children = [c for c in list(ev_el) if isinstance(c.tag, str)]
        if len(body_children) != 1:
            raise ValueError(
                f"<event id='{event_id}'> must contain exactly 1 body element, found {len(body_children)}"
            )
        body_el = body_children[0]
        body = _xml_to_body(body_el)
        events.append(SessionEvent(event_id=event_id, body=body))

    return Session(session_id=session_id, events=events)


def _body_to_xml(parent: ET.Element, body: EventBody) -> None:
    body_tag = body.__class__.__name__
    body_el = ET.SubElement(parent, body_tag)

    if not is_dataclass(body):
        raise TypeError(f"Event body must be a dataclass instance, got {type(body)}")

    payload = asdict(body)
    _dict_to_xml(body_el, payload)


def _dict_to_xml(parent: ET.Element, data: Dict[str, Any]) -> None:
    """
    Convert dict -> nested XML.
    Supports nested dataclass dicts (e.g. ExecResult.output).
    """
    for k, v in data.items():
        child = ET.SubElement(parent, k)
        if isinstance(v, dict):
            _dict_to_xml(child, v)
        elif v is None:
            # Omit text to represent None (optional fields)
            child.text = ""
        else:
            child.text = str(v)


def _xml_to_body(body_el: ET.Element) -> EventBody:
    """
    Convert <HumanMsg>...</HumanMsg> -> HumanMsg(...)
    """
    cls_name = body_el.tag
    cls = _EVENT_TYPE_BY_NAME.get(cls_name)
    if cls is None:
        known = ", ".join(sorted(_EVENT_TYPE_BY_NAME.keys()))
        raise ValueError(f"Unknown event type <{cls_name}>. Known: {known}")

    payload = _xml_to_dict(body_el)
    # Instantiate the dataclass event type
    try:
        return cls(**payload)  # type: ignore[misc]
    except TypeError as e:
        raise ValueError(f"Failed to construct {cls_name} from payload={payload}") from e


def _xml_to_dict(parent: ET.Element) -> Dict[str, Any]:
    """
    Convert children tags into dict recursively.
    Notes:
      - All values come in as strings; we do minimal casting for known primitives.
      - You can extend casting rules as needed.
    """
    out: Dict[str, Any] = {}
    for child in list(parent):
        if len(list(child)) > 0:
            out[child.tag] = _xml_to_dict(child)
        else:
            text = child.text if child.text is not None else ""
            out[child.tag] = _cast_scalar(text)
    return out


def _cast_scalar(s: str) -> Any:
    """
    Best-effort scalar casting:
      - "true"/"false" -> bool
      - ints -> int
      - floats -> float
      - else -> str
    Empty string stays "" (your policy).
    """
    t = s.strip()
    if t.lower() == "true":
        return True
    if t.lower() == "false":
        return False

    # int?
    try:
        if t and t.lstrip("-").isdigit():
            return int(t)
    except Exception:
        pass

    # float?
    try:
        if t and any(c in t for c in [".", "e", "E"]):
            return float(t)
    except Exception:
        pass

    return s


def _indent(elem: ET.Element, level: int = 0) -> None:
    """
    Pretty-print indentation for human-edited XML.
    """
    i = "\n" + ("  " * level)
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            _indent(e, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
