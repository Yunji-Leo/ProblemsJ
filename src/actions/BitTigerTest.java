package actions;

public class BitTigerTest {
    BitTiger test = new BitTiger();

    @org.junit.Test
    public void testMethod() {
        int result = test.largestRectangleArea(new int[]{2, 1, 2});
        test.minWindow("ADOBECODEBANC", "ABC");
        test.search(new int[]{3, 1}, 3);
    }
}
