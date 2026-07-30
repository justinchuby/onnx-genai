// Copyright (c) Microsoft Corporation.
//
// A minimal DOM good enough to unit-test the dashboard's rendering helpers
// under plain `node --test`.
//
// WHY NOT jsdom: this demo has a hard no-npm-install constraint, and the rules
// worth testing here — "an unavailable field must render an em-dash and never a
// number", "the unit survives when the value does not" — need element creation,
// attributes, classes and text content. They do not need layout, CSS cascade or
// event dispatch. A short honest fake buys the coverage that matters without
// putting a dependency tree in front of a contributor.
//
// This is a TEST HELPER. It is never loaded by the browser, and it lives
// outside the `*.test.js` glob so the runner does not treat it as a suite.

class FakeClassList {
  /** @param {FakeElement} owner */
  constructor(owner) {
    this.owner = owner;
    /** @type {Set<string>} */
    this.tokens = new Set();
  }

  /** @param {...string} names */
  add(...names) {
    for (const name of names) {
      // The real DOM throws InvalidCharacterError here. A fake that quietly
      // accepts "a b" turns a browser-breaking bug into a passing test, so this
      // one throws too.
      if (name === '' || /\s/.test(name)) {
        throw new Error(
          `InvalidCharacterError: classList.add() rejects "${name}" — pass separate arguments`,
        );
      }
      this.tokens.add(name);
    }
    this.owner.attributes.class = [...this.tokens].join(' ');
  }

  /** @param {string} name */
  contains(name) {
    return this.tokens.has(name);
  }

  /** @returns {IterableIterator<string>} */
  values() {
    return this.tokens.values();
  }
}

class FakeNode {
  constructor() {
    /** @type {FakeNode[]} */
    this.children = [];
    /** @type {FakeNode|null} */
    this.parent = null;
  }

  /** @returns {string} */
  get textContent() {
    return this.children.map((child) => child.textContent).join('');
  }
}

class FakeText extends FakeNode {
  /** @param {string} text */
  constructor(text) {
    super();
    this.data = String(text);
  }

  get textContent() {
    return this.data;
  }
}

class FakeFragment extends FakeNode {
  /** @param {...FakeNode} nodes */
  append(...nodes) {
    for (const node of nodes) {
      this.children.push(node);
    }
  }
}

class FakeElement extends FakeNode {
  /** @param {string} tagName */
  constructor(tagName) {
    super();
    this.tagName = tagName.toUpperCase();
    /** @type {Record<string, string>} */
    this.attributes = {};
    this.classList = new FakeClassList(this);
    /** @type {string|null} */
    this.ownText = null;
    /** @type {Record<string, Array<(event: unknown) => void>>} */
    this.listeners = {};
  }

  get textContent() {
    if (this.ownText !== null) {
      return this.ownText;
    }
    return this.children.map((child) => child.textContent).join('');
  }

  set textContent(value) {
    this.ownText = String(value);
    this.children = [];
  }

  /** @param {string} name @param {string} value */
  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }

  /** @param {string} name */
  getAttribute(name) {
    return Object.prototype.hasOwnProperty.call(this.attributes, name)
      ? this.attributes[name]
      : null;
  }

  /** @param {string} name */
  hasAttribute(name) {
    return Object.prototype.hasOwnProperty.call(this.attributes, name);
  }

  /** @param {string} name */
  removeAttribute(name) {
    delete this.attributes[name];
  }

  // In a real DOM, `hidden` the IDL property REFLECTS the `hidden` content
  // attribute — setAttribute('hidden', '') makes element.hidden true, and
  // element.hidden = false removes the attribute. Modelling it as a plain field
  // let the two drift apart, which would have hidden a repaint scheduler that
  // kept painting an off-screen panel in the browser while looking correct
  // here.
  get hidden() {
    return this.hasAttribute('hidden');
  }

  set hidden(value) {
    if (value) {
      this.attributes.hidden = '';
    } else {
      delete this.attributes.hidden;
    }
  }

  /** @param {...(FakeNode|string)} nodes */
  append(...nodes) {
    for (const node of nodes) {
      const child = typeof node === 'string' ? new FakeText(node) : node;
      if (child instanceof FakeFragment) {
        for (const grandchild of child.children) {
          grandchild.parent = this;
          this.children.push(grandchild);
        }
        continue;
      }
      if (this.ownText !== null) {
        this.children.push(new FakeText(this.ownText));
        this.ownText = null;
      }
      child.parent = this;
      this.children.push(child);
    }
  }

  /**
   * Detach this node from its parent, mirroring Element.remove(). Modelled
   * because panel code legitimately calls it: without it the fake DOM silently
   * no-ops where the browser detaches, so a test can pass while the real page
   * accumulates duplicate nodes on every repaint.
   */
  remove() {
    if (!this.parent) return;
    const index = this.parent.children.indexOf(this);
    if (index >= 0) this.parent.children.splice(index, 1);
    this.parent = null;
  }

  /** @param {...FakeNode} nodes */
  replaceChildren(...nodes) {
    this.children = [];
    this.ownText = null;
    this.append(...nodes);
  }

  /** @param {string} type @param {(event: unknown) => void} handler */
  addEventListener(type, handler) {
    (this.listeners[type] ??= []).push(handler);
  }

  /**
   * Focus tracking, because AC29/AC30 behaviour is entirely about where focus
   * is and whether it survives. A fake that cannot lose focus cannot show that
   * the real one does.
   */
  focus() {
    if (globalThis.document) {
      globalThis.document.activeElement = this;
    }
  }

  blur() {
    if (globalThis.document?.activeElement === this) {
      globalThis.document.activeElement = null;
    }
  }

  /**
   * @param {FakeNode|null} node
   * @returns {boolean} Whether node is this element or a descendant of it.
   */
  contains(node) {
    for (let cursor = node; cursor; cursor = cursor.parent) {
      if (cursor === this) return true;
    }
    return false;
  }

  get ownerDocument() {
    return globalThis.document ?? null;
  }

  /**
   * Dispatch an event, bubbling to ancestors the way a real one does. Roving
   * focus relies on delegation from a container that outlives its children, so
   * a non-bubbling fake would make the design look broken.
   *
   * @param {{type: string, [key: string]: unknown}} event
   */
  dispatchEvent(event) {
    const dispatched = { defaultPrevented: false, ...event };
    dispatched.target ??= this;
    dispatched.preventDefault = () => {
      dispatched.defaultPrevented = true;
    };
    for (let cursor = this; cursor; cursor = cursor.parent) {
      dispatched.currentTarget = cursor;
      for (const handler of cursor.listeners?.[event.type] ?? []) {
        handler(dispatched);
      }
    }
    return !dispatched.defaultPrevented;
  }

  /**
   * Canvas panels call this. Returning null exercises the painter's
   * no-context early return, which is the same path a browser takes when a
   * context is unavailable — so panel tests cover the guard for free.
   *
   * @returns {null}
   */
  getContext() {
    return null;
  }

  /** @param {string} type */
  removeEventListener(type) {
    delete this.listeners[type];
  }

  /**
   * Depth-first search by class name. Enough for assertions; not a CSS engine.
   *
   * @param {string} className
   * @returns {FakeElement|null}
   */
  findByClass(className) {
    for (const child of this.children) {
      if (child instanceof FakeElement) {
        if (child.classList.contains(className)) {
          return child;
        }
        const found = child.findByClass(className);
        if (found) {
          return found;
        }
      }
    }
    return null;
  }

  /**
   * @param {string} tagName
   * @returns {FakeElement[]}
   */
  findAllByTag(tagName) {
    const wanted = tagName.toUpperCase();
    /** @type {FakeElement[]} */
    const found = [];
    for (const child of this.children) {
      if (child instanceof FakeElement) {
        if (child.tagName === wanted) {
          found.push(child);
        }
        found.push(...child.findAllByTag(tagName));
      }
    }
    return found;
  }
}

/** @type {Array<(time: number) => void>} */
const pendingFrames = [];

/**
 * Run every animation frame callback queued so far.
 *
 * The fake `requestAnimationFrame` QUEUES rather than running synchronously,
 * because synchronous frames would hide exactly the bug this scheduler exists
 * to prevent: a callback that runs before its handle is assigned. Tests decide
 * when frames happen, which is deterministic without being unrealistic.
 *
 * @returns {number} How many callbacks ran.
 */
export function flushAnimationFrames() {
  const queued = pendingFrames.splice(0, pendingFrames.length);
  for (const callback of queued) {
    callback(0);
  }
  return queued.length;
}

/**
 * Install the fake DOM on `globalThis` so modules written for the browser run
 * unmodified. Returns a function that removes it again, so one test file cannot
 * leak globals into the next.
 *
 * @returns {() => void}
 */
export function installFakeDom() {
  const previousDocument = globalThis.document;
  const previousRaf = globalThis.requestAnimationFrame;
  const previousCancelRaf = globalThis.cancelAnimationFrame;

  pendingFrames.length = 0;
  globalThis.document = {
    activeElement: null,
    createElement: (tagName) => new FakeElement(tagName),
    createTextNode: (text) => new FakeText(text),
    createDocumentFragment: () => new FakeFragment(),
  };
  globalThis.requestAnimationFrame = (callback) => {
    pendingFrames.push(callback);
    return pendingFrames.length;
  };
  globalThis.cancelAnimationFrame = (handle) => {
    const index = handle - 1;
    if (index >= 0 && index < pendingFrames.length) {
      pendingFrames[index] = () => {};
    }
  };

  return () => {
    pendingFrames.length = 0;
    globalThis.document = previousDocument;
    globalThis.requestAnimationFrame = previousRaf;
    globalThis.cancelAnimationFrame = previousCancelRaf;
  };
}

export { FakeElement, FakeFragment, FakeText };
