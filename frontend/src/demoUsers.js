export const TEST_USERS = Object.freeze([
  Object.freeze({
    id: "talal",
    name: "Talal Sabbagh",
    label: "Talal Sabbagh - Employee",
    role: "Employee",
    email: "talal.sabbagh@example.com",
    instance: "portalv21",
  }),
  Object.freeze({
    id: "hana",
    name: "Hana Haddad",
    label: "Hana Haddad - HOD",
    role: "HOD",
    email: "hana.haddad@example.com",
    instance: "portalv21",
  }),
]);

export const DEFAULT_TEST_USER = TEST_USERS[0];

export function findTestUser(userId) {
  return TEST_USERS.find((user) => user.id === userId) || DEFAULT_TEST_USER;
}
