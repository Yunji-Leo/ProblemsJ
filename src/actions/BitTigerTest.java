package actions;

public class BitTigerTest {
    BitTiger test = new BitTiger();

    @org.junit.Test
    public void testMethod() {
        test.minWindow("ADOBECODEBANC", "ABC");
        test.search(new int[]{3, 1}, 3);
    }
}
